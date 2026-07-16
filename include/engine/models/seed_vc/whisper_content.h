#pragma once

#include "engine/framework/modules/whisper_embedding.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcWhisperEncoderWeights;

class SeedVcWhisperContentEncoder {
public:
    SeedVcWhisperContentEncoder() = default;
    explicit SeedVcWhisperContentEncoder(std::shared_ptr<const SeedVcWhisperEncoderWeights> weights);
    ~SeedVcWhisperContentEncoder();

    SeedVcWhisperContentEncoder(SeedVcWhisperContentEncoder &&) noexcept;
    SeedVcWhisperContentEncoder & operator=(SeedVcWhisperContentEncoder &&) noexcept;
    SeedVcWhisperContentEncoder(const SeedVcWhisperContentEncoder &) = delete;
    SeedVcWhisperContentEncoder & operator=(const SeedVcWhisperContentEncoder &) = delete;

    int64_t channels() const noexcept;
    std::vector<float> extract_16k_mono(
        const std::vector<float> & waveform_16k,
        size_t threads) const;

private:
    struct State;

    std::shared_ptr<const SeedVcWhisperEncoderWeights> weights_;
    engine::modules::WhisperEmbeddingConfig config_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
