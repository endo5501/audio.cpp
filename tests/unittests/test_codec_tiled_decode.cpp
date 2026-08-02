// Irodori-TTS codec デコードのタイル分割テスト。
//
// 目的は 2 つ:
//   1. タイル分割の窓/トリム計算が正しいこと (合成 decode_window による厳密な検査)
//   2. 実際の DACVAE デコーダ構造において、タイル分割の出力が分割なしの出力と
//      一致すること (合成重みを使うのでモデルファイルは不要)
//
// 2 は design.md D1 の「デコーダは Snake1d / Conv1d / ConvTranspose1d / Slice / Add
// のみで構成され、受容野は片側 8 latent フレーム。オーバーラップがそれ以上なら
// 分割は厳密」という主張をそのまま assert にしたものである。

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/module.h"
#include "engine/models/irodori_tts/codec.h"
#include "test_assert.h"

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using engine::test::require;
using engine::test::require_eq;

namespace ir = engine::models::irodori_tts;
namespace core = engine::core;

// ---------------------------------------------------------------------------
// 2.3 窓/トリム計算 — 合成 decode_window による厳密な検査
// ---------------------------------------------------------------------------

struct WindowCall {
    int64_t start = 0;
    int64_t frames = 0;
};

// decode_window がグローバル位置 x に対して値 x を返すなら、タイル分割の結果は
// 0,1,2,...,frames*upsample-1 でなければならない。トリム位置が 1 サンプルでも
// ずれれば即座に検出できる。
std::vector<float> run_identity_tiling(
    int64_t frames,
    int64_t tile_frames,
    int64_t overlap_frames,
    int64_t upsample,
    std::vector<WindowCall> & calls) {
    calls.clear();
    return ir::irodori_codec_tiled_decode(
        frames,
        tile_frames,
        overlap_frames,
        upsample,
        [&](int64_t win_start, int64_t win_frames) {
            calls.push_back({win_start, win_frames});
            std::vector<float> out(static_cast<size_t>(win_frames * upsample));
            for (int64_t i = 0; i < win_frames * upsample; ++i) {
                out[static_cast<size_t>(i)] = static_cast<float>(win_start * upsample + i);
            }
            return out;
        });
}

void check_identity(
    const char * label,
    int64_t frames,
    int64_t tile_frames,
    int64_t overlap_frames,
    int64_t upsample) {
    std::vector<WindowCall> calls;
    const auto out = run_identity_tiling(frames, tile_frames, overlap_frames, upsample, calls);

    require_eq(static_cast<int64_t>(out.size()), frames * upsample,
               std::string(label) + " output sample count");

    for (int64_t i = 0; i < frames * upsample; ++i) {
        if (out[static_cast<size_t>(i)] != static_cast<float>(i)) {
            std::ostringstream oss;
            oss << label << " sample " << i << " mismatch: expected " << i
                << " actual " << out[static_cast<size_t>(i)];
            throw std::runtime_error(oss.str());
        }
    }

    // メモリ上界の保証: どの窓も tile_frames を超えてはならない。
    // これが破れるとタイル分割の存在意義そのものが失われる。
    require(!calls.empty(), std::string(label) + " must decode at least one window");
    for (const auto & call : calls) {
        if (call.frames > tile_frames) {
            std::ostringstream oss;
            oss << label << " window at " << call.start << " has " << call.frames
                << " frames, exceeding tile " << tile_frames;
            throw std::runtime_error(oss.str());
        }
        require(call.frames > 0, std::string(label) + " window must be non-empty");
    }

    std::cout << "[ok] " << label << " frames=" << frames << " tile=" << tile_frames
              << " overlap=" << overlap_frames << " windows=" << calls.size() << '\n';
}

void test_tiled_decode_index_math() {
    constexpr int64_t kUpsample = 8;

    // (a) frames <= tile: 分割せず 1 回だけ、窓は [0, frames)
    {
        std::vector<WindowCall> calls;
        const auto out = run_identity_tiling(64, 64, 16, kUpsample, calls);
        require_eq(static_cast<int64_t>(out.size()), 64 * kUpsample, "direct path sample count");
        require_eq(static_cast<size_t>(calls.size()), static_cast<size_t>(1),
                   "direct path window count");
        require_eq(calls[0].start, static_cast<int64_t>(0), "direct path window start");
        require_eq(calls[0].frames, static_cast<int64_t>(64), "direct path window frames");
        std::cout << "[ok] frames == tile takes the direct path (1 window)\n";
    }
    {
        std::vector<WindowCall> calls;
        run_identity_tiling(30, 64, 16, kUpsample, calls);
        require_eq(static_cast<size_t>(calls.size()), static_cast<size_t>(1),
                   "short latent window count");
        require_eq(calls[0].frames, static_cast<int64_t>(30), "short latent window frames");
        std::cout << "[ok] frames < tile takes the direct path (1 window)\n";
    }

    // (b) frames が stride の倍数
    check_identity("stride multiple", 128, 64, 16, kUpsample);   // stride 32
    // (c) 端数タイルが出る
    check_identity("fractional tail", 100, 64, 16, kUpsample);
    check_identity("long latent", 200, 64, 16, kUpsample);
    check_identity("tail of one frame", 65, 64, 16, kUpsample);
    // (d) 既定に近い比率
    check_identity("default-like ratio", 498, 256, 16, kUpsample);
}

// ---------------------------------------------------------------------------
// 2.4 タイル設定のバリデーション
// ---------------------------------------------------------------------------

bool throws_runtime_error(int64_t tile_frames, int64_t overlap_frames) {
    try {
        ir::validate_irodori_codec_decode_tiling(tile_frames, overlap_frames);
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

// irodori_codec_tiled_decode 自体が設定を検査するかどうか。
bool tiled_decode_throws(int64_t frames, int64_t tile_frames, int64_t overlap_frames) {
    try {
        ir::irodori_codec_tiled_decode(
            frames, tile_frames, overlap_frames, 8,
            [](int64_t, int64_t win_frames) {
                return std::vector<float>(static_cast<size_t>(win_frames * 8), 0.0f);
            });
    } catch (const std::runtime_error &) {
        return true;
    }
    return false;
}

void test_tiling_validation() {
    require(throws_runtime_error(32, 16), "overlap*2 == tile must be rejected");
    require(throws_runtime_error(32, 20), "overlap*2 > tile must be rejected");
    require(throws_runtime_error(256, 7), "overlap below the receptive field must be rejected");
    require(throws_runtime_error(256, 0), "zero overlap must be rejected");
    require(throws_runtime_error(0, 16), "non-positive tile must be rejected");
    require(!throws_runtime_error(256, 16), "the documented default must be accepted");
    require(!throws_runtime_error(512, 16), "the Metal default must be accepted");
    require(!throws_runtime_error(32, 8), "overlap equal to the receptive field must be accepted");

    // 設定の妥当性は frames に依存してはならない。frames <= tile の直接経路でも
    // 検査されなければ、「短い発話では通り、長い発話で初めて落ちる」設定エラーを
    // 作り込むことになる。
    require(tiled_decode_throws(300, 256, 0),
            "invalid overlap must be rejected when tiling");
    require(tiled_decode_throws(200, 256, 0),
            "invalid overlap must be rejected on the direct path too");
    require(tiled_decode_throws(200, 256, 7),
            "overlap below the receptive field must be rejected on the direct path too");
    require(!tiled_decode_throws(200, 256, 16),
            "a valid configuration must not be rejected on the direct path");
    std::cout << "[ok] tiling configuration validation\n";
}

// ---------------------------------------------------------------------------
// 2.5 バックエンド別の既定タイルサイズ
// ---------------------------------------------------------------------------

void test_default_tile_frames() {
    require_eq(ir::irodori_codec_default_decode_tile_frames(core::BackendType::Vulkan),
               static_cast<int64_t>(256), "Vulkan default tile");
    require_eq(ir::irodori_codec_default_decode_tile_frames(core::BackendType::Metal),
               static_cast<int64_t>(512), "Metal default tile");
    require_eq(ir::irodori_codec_default_decode_tile_frames(core::BackendType::Cuda),
               static_cast<int64_t>(512), "CUDA default tile");
    require_eq(ir::irodori_codec_default_decode_tile_frames(core::BackendType::Cpu),
               static_cast<int64_t>(512), "CPU default tile");
    std::cout << "[ok] backend-specific default tile sizes\n";
}

void test_resolve_tile_frames() {
    // 未指定ならバックエンド既定に落ちる。
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(std::nullopt,
                                                            core::BackendType::Vulkan),
               static_cast<int64_t>(256), "Vulkan resolves to its default");
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(std::nullopt,
                                                            core::BackendType::Metal),
               static_cast<int64_t>(512), "Metal resolves to its default");
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(std::nullopt,
                                                            core::BackendType::Cpu),
               static_cast<int64_t>(512), "CPU resolves to its default");

    // 明示指定はバックエンドに関わらず優先される。既定より小さくても大きくてもよい。
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(128, core::BackendType::Vulkan),
               static_cast<int64_t>(128), "explicit tile overrides the Vulkan default");
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(128, core::BackendType::Metal),
               static_cast<int64_t>(128), "explicit tile overrides the Metal default");
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(1024, core::BackendType::Cpu),
               static_cast<int64_t>(1024), "explicit tile larger than the default is honoured");
    require_eq(ir::irodori_codec_resolve_decode_tile_frames(256, core::BackendType::Cuda),
               static_cast<int64_t>(256), "explicit tile equal to another backend's default");
    std::cout << "[ok] explicit tile size overrides the backend default\n";
}

// ---------------------------------------------------------------------------
// 2.2 実デコーダ構造での一致 (合成重み)
// ---------------------------------------------------------------------------

constexpr int64_t kUpsampleFactor = 1920;  // rates {12,10,8,2} は builder 側で固定
constexpr size_t kGraphArenaBytes = 256ull * 1024ull * 1024ull;
constexpr size_t kWeightContextBytes = 64ull * 1024ull * 1024ull;

std::vector<float> patterned(size_t count, float phase, float scale) {
    std::vector<float> values(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        const float x = static_cast<float>(i);
        values[i] = scale * (std::sin(phase + 0.113f * x) + 0.5f * std::cos(phase * 0.7f + 0.071f * x));
    }
    return values;
}

struct GgmlContextDeleter {
    void operator()(ggml_context * ctx) const noexcept {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

// 実デコーダと同じ形状の重みを合成値で構築する。
class SyntheticCodec {
public:
    SyntheticCodec(ggml_backend_t backend, const ir::IrodoriCodecConfig & config)
        : backend_(backend), config_(config) {
        weights_.store = std::make_shared<core::BackendWeightStore>(
            backend, core::BackendType::Cpu, "test.irodori.codec.weights", kWeightContextBytes);
        auto & store = *weights_.store;

        float phase = 0.11f;
        auto next_phase = [&]() {
            phase += 0.37f;
            return phase;
        };

        auto make_conv = [&](int64_t out_ch, int64_t in_ch, int64_t kernel) {
            engine::modules::Conv1dWeights w;
            const float scale = 0.6f / std::sqrt(static_cast<float>(in_ch * kernel));
            w.weight = store.make_f32(
                core::TensorShape::from_dims({out_ch, in_ch, kernel}),
                patterned(static_cast<size_t>(out_ch * in_ch * kernel), next_phase(), scale));
            w.bias = store.make_f32(
                core::TensorShape::from_dims({out_ch}),
                patterned(static_cast<size_t>(out_ch), next_phase(), 0.05f));
            return w;
        };
        auto make_conv_transpose = [&](int64_t in_ch, int64_t out_ch, int64_t kernel) {
            engine::modules::ConvTranspose1dWeights w;
            const float scale = 0.6f / std::sqrt(static_cast<float>(in_ch * kernel));
            w.weight = store.make_f32(
                core::TensorShape::from_dims({in_ch, out_ch, kernel}),
                patterned(static_cast<size_t>(in_ch * out_ch * kernel), next_phase(), scale));
            w.bias = store.make_f32(
                core::TensorShape::from_dims({out_ch}),
                patterned(static_cast<size_t>(out_ch), next_phase(), 0.05f));
            return w;
        };
        auto make_snake = [&](int64_t channels) {
            engine::modules::Snake1dWeights w;
            // Snake は alpha が 0 に近いと発散するので 1 の周りに散らす。
            auto alpha = patterned(static_cast<size_t>(channels), next_phase(), 0.2f);
            for (auto & value : alpha) {
                value += 1.0f;
            }
            w.alpha = store.make_f32(core::TensorShape::from_dims({channels}), std::move(alpha));
            return w;
        };
        auto make_residual = [&](int64_t channels) {
            ir::IrodoriCodecResidualUnitWeights w;
            w.snake0 = make_snake(channels);
            w.conv0 = make_conv(channels, channels, 7);
            w.snake1 = make_snake(channels);
            w.conv1 = make_conv(channels, channels, 1);
            return w;
        };

        weights_.quantizer_out_proj = make_conv(config_.latent_dim, config_.codebook_dim, 1);
        weights_.decoder_input = make_conv(config_.decoder_dim, config_.latent_dim, 7);

        const int64_t rates[] = {12, 10, 8, 2};
        int64_t in_channels = config_.decoder_dim;
        for (int64_t block = 0; block < 4; ++block) {
            const int64_t out_channels = in_channels / 2;
            ir::IrodoriCodecDecoderBlockWeights w;
            w.up_snake = make_snake(in_channels);
            w.up_conv = make_conv_transpose(in_channels, out_channels, 2 * rates[block]);
            w.residual_0 = make_residual(out_channels);
            w.residual_1 = make_residual(out_channels);
            w.residual_2 = make_residual(out_channels);
            weights_.decoder_blocks.push_back(std::move(w));
            in_channels = out_channels;
        }
        weights_.watermark_passthrough.snake = make_snake(in_channels);
        weights_.watermark_passthrough.conv = make_conv(1, in_channels, 7);

        store.upload();
    }

    // latent の [win_start, win_start + win_frames) を復号し、
    // win_frames * 1920 サンプルを返す。
    std::vector<float> decode_window(
        const std::vector<float> & latent,
        int64_t win_start,
        int64_t win_frames) const {
        ggml_init_params params{kGraphArenaBytes, nullptr, true};
        std::unique_ptr<ggml_context, GgmlContextDeleter> ctx(ggml_init(params));
        require(ctx != nullptr, "failed to init test decode context");

        core::ModuleBuildContext build_ctx{ctx.get(), "test.irodori.codec_decode",
                                           core::BackendType::Cpu};
        auto latent_value = core::make_tensor(
            build_ctx, GGML_TYPE_F32,
            core::TensorShape::from_dims({1, win_frames, config_.codebook_dim}));
        ggml_set_input(latent_value.tensor);

        auto output = ir::build_irodori_codec_decode(build_ctx, latent_value, weights_, config_);
        output = core::ensure_backend_addressable_layout(build_ctx, output);
        ggml_set_output(output.tensor);

        ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 131072, false);
        ggml_build_forward_expand(graph, output.tensor);

        ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
        require(allocr != nullptr, "failed to create test graph allocator");
        const bool allocated = ggml_gallocr_reserve(allocr, graph) &&
                               ggml_gallocr_alloc_graph(allocr, graph);
        if (!allocated) {
            ggml_gallocr_free(allocr);
            throw std::runtime_error("failed to allocate test codec decode graph");
        }

        std::vector<float> window_latent(
            latent.begin() + static_cast<std::ptrdiff_t>(win_start * config_.codebook_dim),
            latent.begin() +
                static_cast<std::ptrdiff_t>((win_start + win_frames) * config_.codebook_dim));
        core::write_tensor_f32(latent_value, window_latent);

        const ggml_status status = ggml_backend_graph_compute(backend_, graph);
        if (status != GGML_STATUS_SUCCESS) {
            ggml_gallocr_free(allocr);
            throw std::runtime_error("test codec decode graph compute failed");
        }
        auto samples = core::read_tensor_f32(output.tensor);
        ggml_gallocr_free(allocr);

        require_eq(static_cast<int64_t>(samples.size()), win_frames * kUpsampleFactor,
                   "decoded window sample count");
        return samples;
    }

private:
    ggml_backend_t backend_ = nullptr;
    ir::IrodoriCodecConfig config_{};
    ir::IrodoriCodecWeights weights_{};
};

void test_tiled_matches_direct_real_decoder() {
    ir::IrodoriCodecConfig config;
    config.codebook_dim = 4;
    config.latent_dim = 8;
    config.decoder_dim = 32;  // -> 16, 8, 4, 2
    config.hop_length = kUpsampleFactor;
    config.sample_rate = 48000;

    constexpr int64_t kFrames = 100;

    core::BackendConfig backend_config{core::BackendType::Cpu, 0, 4};
    ggml_backend_t backend = core::init_backend(backend_config);
    require(backend != nullptr, "failed to init CPU backend");

    try {
        const SyntheticCodec codec(backend, config);
        const auto latent = patterned(
            static_cast<size_t>(kFrames * config.codebook_dim), 0.29f, 1.0f);

        const auto direct = codec.decode_window(latent, 0, kFrames);

        // 出力は最後に tanh を通るので飽和しうる。全部 +-1 に張り付いていると
        // 一致テストが自明に通ってしまうため、有意な信号であることを確かめる。
        int64_t unsaturated = 0;
        for (float value : direct) {
            if (std::fabs(value) < 0.99f) {
                ++unsaturated;
            }
        }
        require(unsaturated * 10 > static_cast<int64_t>(direct.size()),
                "direct decode output is saturated; the parity check would be vacuous");

        struct ParityCase {
            int64_t tile;
            int64_t overlap;
            const char * note;
        };
        // 2 件目はオーバーラップを受容野ちょうどに置く。この境界に余裕は無い:
        // オーバーラップを 7 に下げると max_diff が 0 から 5.96e-08 (1 ULP) に
        // 変わることを実測で確認している。デコーダの構造が変わって受容野が 9 以上に
        // なれば、ここが最初に壊れる。定数をコメントではなくテストで固定するのが狙い。
        const ParityCase cases[] = {
            {48, 16, "default-like overlap"},
            {32, ir::kIrodoriCodecDecodeReceptiveFieldFrames, "overlap == receptive field"},
        };

        for (const auto & parity : cases) {
            const auto tiled = ir::irodori_codec_tiled_decode(
                kFrames, parity.tile, parity.overlap, kUpsampleFactor,
                [&](int64_t win_start, int64_t win_frames) {
                    return codec.decode_window(latent, win_start, win_frames);
                });

            require_eq(static_cast<int64_t>(tiled.size()), static_cast<int64_t>(direct.size()),
                       "tiled vs direct sample count");

            // 許容誤差は置かない。オーバーラップが受容野以上なら、各出力サンプルは
            // 分割の有無に関わらず同じ入力に同じ順序で同じ演算を適用した結果になり、
            // ビット単位で一致する。1 ULP でも許すと、受容野が不足したときの
            // 破綻 (実測 5.96e-08) を見逃す。
            float max_diff = 0.0f;
            int64_t worst_index = -1;
            for (size_t i = 0; i < direct.size(); ++i) {
                const float diff = std::fabs(direct[i] - tiled[i]);
                if (diff > max_diff) {
                    max_diff = diff;
                    worst_index = static_cast<int64_t>(i);
                }
            }
            if (max_diff != 0.0f) {
                std::ostringstream oss;
                oss << "tiled decode is not bit-exact vs direct decode (" << parity.note
                    << ", tile=" << parity.tile << " overlap=" << parity.overlap
                    << "): max_diff=" << max_diff << " at sample " << worst_index
                    << " (direct=" << direct[static_cast<size_t>(worst_index)]
                    << " tiled=" << tiled[static_cast<size_t>(worst_index)] << ')';
                throw std::runtime_error(oss.str());
            }

            std::cout << "[ok] tiled decode is bit-exact vs direct decode on the real decoder"
                      << " frames=" << kFrames << " tile=" << parity.tile
                      << " overlap=" << parity.overlap << " (" << parity.note << ')'
                      << " samples=" << direct.size()
                      << " unsaturated=" << unsaturated << '\n';
        }
    } catch (...) {
        ggml_backend_free(backend);
        throw;
    }
    ggml_backend_free(backend);
}

}  // namespace

int main() {
    try {
        test_default_tile_frames();
        test_resolve_tile_frames();
        test_tiling_validation();
        test_tiled_decode_index_math();
        test_tiled_matches_direct_real_decoder();
    } catch (const std::exception & ex) {
        std::cerr << "[FAIL] " << ex.what() << '\n';
        return 1;
    }
    std::cout << "[PASS] codec tiled decode\n";
    return 0;
}
