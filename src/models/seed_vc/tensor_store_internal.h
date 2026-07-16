#pragma once

#include "engine/framework/core/backend.h"
#include "engine/framework/core/backend_weight_store.h"
#include "engine/framework/core/execution_context.h"
#include "engine/framework/core/module.h"
#include "engine/framework/assets/tensor_source.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine::models::seed_vc {

struct SeedVcTensorStore {
    std::filesystem::path source_path;
    std::string name;
    std::shared_ptr<engine::core::ExecutionContext> execution_context;
    std::shared_ptr<engine::core::BackendWeightStore> store;
    std::unordered_map<std::string, engine::core::TensorValue> tensors;
    int64_t loaded_tensor_count = 0;
    int64_t skipped_generated_constant_count = 0;
    int64_t parameter_count = 0;
};

struct SeedVcV2ArWeights final : SeedVcTensorStore {};
struct SeedVcV2CfmWeights final : SeedVcTensorStore {};
struct SeedVcV1ModelWeights final : SeedVcTensorStore {};
struct SeedVcAstralWeights final : SeedVcTensorStore {};
struct SeedVcRmvpeWeights final : SeedVcTensorStore {};
struct SeedVcWhisperEncoderWeights final : SeedVcTensorStore {};

void load_seed_vc_tensor_store(
    SeedVcTensorStore & weights,
    std::shared_ptr<const engine::assets::TensorSource> source,
    const std::string & name,
    engine::core::BackendConfig backend,
    const std::vector<std::string> & generated_constants,
    const std::vector<std::string> & generated_constant_suffixes = {},
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::Native);

template <typename Weights>
std::shared_ptr<const Weights> load_seed_vc_checkpoint(
    std::shared_ptr<const engine::assets::TensorSource> source,
    const std::string & name,
    engine::core::BackendConfig backend,
    const std::vector<std::string> & generated_constants,
    const std::vector<std::string> & generated_constant_suffixes = {},
    engine::assets::TensorStorageType storage_type = engine::assets::TensorStorageType::Native) {
    auto weights = std::make_shared<Weights>();
    load_seed_vc_tensor_store(
        *weights,
        std::move(source),
        name,
        std::move(backend),
        generated_constants,
        generated_constant_suffixes,
        storage_type);
    return weights;
}

}  // namespace engine::models::seed_vc
