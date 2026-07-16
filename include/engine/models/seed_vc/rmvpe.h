#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcRmvpeWeights;

class SeedVcRmvpeF0Extractor {
public:
    SeedVcRmvpeF0Extractor() = default;
    explicit SeedVcRmvpeF0Extractor(std::shared_ptr<const SeedVcRmvpeWeights> weights);
    ~SeedVcRmvpeF0Extractor();

    SeedVcRmvpeF0Extractor(SeedVcRmvpeF0Extractor &&) noexcept;
    SeedVcRmvpeF0Extractor & operator=(SeedVcRmvpeF0Extractor &&) noexcept;
    SeedVcRmvpeF0Extractor(const SeedVcRmvpeF0Extractor &) = delete;
    SeedVcRmvpeF0Extractor & operator=(const SeedVcRmvpeF0Extractor &) = delete;

    std::vector<float> infer_16k_mono(
        const std::vector<float> & waveform_16k,
        float threshold,
        size_t threads) const;

private:
    struct State;

    std::shared_ptr<const SeedVcRmvpeWeights> weights_;
    std::shared_ptr<State> state_;
};

}  // namespace engine::models::seed_vc
