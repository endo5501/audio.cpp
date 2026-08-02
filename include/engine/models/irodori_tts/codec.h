#pragma once

#include "engine/framework/assets/tensor_source.h"
#include "engine/framework/core/backend.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/modules/activation_modules.h"
#include "engine/framework/modules/conv_modules.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/irodori_tts/assets.h"

#include <ggml-backend.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace engine::core {
class BackendWeightStore;
}

namespace engine::models::irodori_tts {

struct IrodoriCodecResidualUnitWeights {
  modules::Snake1dWeights snake0;
  modules::Conv1dWeights conv0;
  modules::Snake1dWeights snake1;
  modules::Conv1dWeights conv1;
};

struct IrodoriCodecDecoderBlockWeights {
  modules::Snake1dWeights up_snake;
  modules::ConvTranspose1dWeights up_conv;
  IrodoriCodecResidualUnitWeights residual_0;
  IrodoriCodecResidualUnitWeights residual_1;
  IrodoriCodecResidualUnitWeights residual_2;
};

struct IrodoriCodecEncoderBlockWeights {
  IrodoriCodecResidualUnitWeights residual_0;
  IrodoriCodecResidualUnitWeights residual_1;
  IrodoriCodecResidualUnitWeights residual_2;
  modules::Snake1dWeights down_snake;
  modules::Conv1dWeights down_conv;
};

struct IrodoriCodecWatermarkPassthroughWeights {
  modules::Snake1dWeights snake;
  modules::Conv1dWeights conv;
};

struct IrodoriCodecWeights {
  std::shared_ptr<core::BackendWeightStore> store;
  modules::Conv1dWeights encoder_input;
  std::vector<IrodoriCodecEncoderBlockWeights> encoder_blocks;
  modules::Snake1dWeights encoder_output_snake;
  modules::Conv1dWeights encoder_output;
  modules::Conv1dWeights quantizer_in_proj;
  modules::Conv1dWeights quantizer_out_proj;
  modules::Conv1dWeights decoder_input;
  std::vector<IrodoriCodecDecoderBlockWeights> decoder_blocks;
  IrodoriCodecWatermarkPassthroughWeights watermark_passthrough;
};

IrodoriCodecWeights
load_irodori_codec_weights(const IrodoriTTSAssets &assets, ggml_backend_t backend,
                           core::BackendType backend_type,
                           size_t weight_context_bytes,
                           assets::TensorStorageType conv_storage_type);

core::TensorValue build_irodori_codec_decode(
    core::ModuleBuildContext &ctx, const core::TensorValue &latent_btd,
    const IrodoriCodecWeights &weights, const IrodoriCodecConfig &config);

core::TensorValue build_irodori_codec_encode(
    core::ModuleBuildContext &ctx, const core::TensorValue &waveform_bct,
    const IrodoriCodecWeights &weights, const IrodoriCodecConfig &config);

// デコーダの受容野 (片側, latent フレーム数)。
//
// 出力インデックスを層ごとに逆伝播して求める。窓を [0, N)、左のトリム量を ov と
// すると、残す先頭サンプルは 1920*ov。ConvTranspose は kernel=2s / stride=s /
// 両端 p=(s+1)/2 の切り落としなので、出力 j は入力 floor((j+p)/s)-1 以上を必要とする。
//
//   watermark Conv1d(k7,p3)      1920ov - 3
//   block4 residual (d=1/3/9)    1920ov - 42        (3+9+27=39)
//   block4 ConvT (s2,p1)         960ov - 22
//   block3 residual              960ov - 61
//   block3 ConvT (s8,p4)         120ov - 9
//   block2 residual              120ov - 48
//   block2 ConvT (s10,p5)        12ov - 6
//   block1 residual              12ov - 45
//   block1 ConvT (s12,p6)        ov - 5
//   decoder_input Conv1d(k7,p3)  ov - 8
//
// これが 0 以上であること、すなわち **ov >= 8**。右端も同様に (N-ov)+7 <= N-1 から
// 同じ条件になる。等号成立で余裕は無く、ov=7 ではビット一致が崩れる (実測 1 ULP)。
// この値は test_codec_tiled_decode.cpp の parity ケースで固定してある。
inline constexpr int64_t kIrodoriCodecDecodeReceptiveFieldFrames = 8;

// バックエンド別の既定タイルサイズ。Vulkan は一部ドライバ (AMD) が単一バッファを
// 2 GiB に制限するため小さく取る。CUDA / CPU には上限が無く、Metal は超過分を
// ビュー分割で吸収するため、より大きなタイルを安全に使える。
int64_t
irodori_codec_default_decode_tile_frames(core::BackendType backend_type) noexcept;

// タイル設定の妥当性を検証する。不正なら std::runtime_error を送出する。
void validate_irodori_codec_decode_tiling(int64_t tile_frames,
                                          int64_t overlap_frames);

// decode_window(win_start, win_frames) は latent の当該範囲を復号し、
// win_frames * upsample サンプルを返さなければならない。
using IrodoriCodecDecodeWindowFn =
    std::function<std::vector<float>(int64_t win_start, int64_t win_frames)>;

// latent フレーム軸をオーバーラップ付きタイルに分割して復号し、各タイルの
// オーバーラップ領域をトリムして連結する。frames <= tile_frames の場合は
// 分割せず decode_window を 1 回だけ呼ぶ。
// どの窓も tile_frames フレームを超えないため、計算グラフの単一テンソルサイズは
// 発話長ではなくタイルサイズによって決まる。
std::vector<float> irodori_codec_tiled_decode(
    int64_t frames, int64_t tile_frames, int64_t overlap_frames,
    int64_t upsample, const IrodoriCodecDecodeWindowFn &decode_window);

// デコードのタイル設定。tile_frames が未指定ならバックエンド既定を使う。
struct IrodoriCodecDecodeTiling {
  std::optional<int64_t> tile_frames;
  int64_t overlap_frames = 16;
};

class IrodoriCodec {
public:
  IrodoriCodec(std::shared_ptr<const IrodoriTTSAssets> assets,
               core::ExecutionContext &execution_context,
               size_t graph_arena_bytes, size_t weight_context_bytes,
               assets::TensorStorageType weight_storage_type,
               IrodoriCodecDecodeTiling decode_tiling = {});
  ~IrodoriCodec();

  runtime::AudioBuffer decode(const std::vector<float> &latent,
                              int64_t latent_steps, int64_t target_samples);
  std::vector<float> encode_reference(const runtime::AudioBuffer &audio,
                                      int64_t &latent_steps_out);
  void release_graphs();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace engine::models::irodori_tts
