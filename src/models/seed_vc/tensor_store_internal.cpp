#include "tensor_store_internal.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace engine::models::seed_vc {
namespace {

bool is_float_dtype(const std::string & dtype) {
    return dtype == "F32" || dtype == "F16" || dtype == "BF16";
}

int64_t tensor_elements(const std::vector<int64_t> & shape, const std::string & tensor_name) {
    if (shape.empty()) {
        throw std::runtime_error("Seed-VC tensor shape is empty: " + tensor_name);
    }
    return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [&](int64_t lhs, int64_t rhs) {
        if (rhs <= 0) {
            throw std::runtime_error("Seed-VC tensor shape contains non-positive dimension: " + tensor_name);
        }
        return lhs * rhs;
    });
}

bool contains_name(const std::vector<std::string> & names, const std::string & name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

bool has_suffix(const std::string & value, const std::string & suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains_suffix(const std::vector<std::string> & suffixes, const std::string & name) {
    return std::any_of(suffixes.begin(), suffixes.end(), [&](const std::string & suffix) {
        return has_suffix(name, suffix);
    });
}

engine::assets::TensorStorageType storage_type_for_seed_vc_tensor(
    const std::string & name,
    const std::vector<int64_t> & shape,
    engine::assets::TensorStorageType requested_storage_type) {
    if (shape.size() <= 1 ||
        name.find("embed") != std::string::npos ||
        name.find("embedding") != std::string::npos ||
        has_suffix(name, ".bias") ||
        has_suffix(name, ".gamma") ||
        has_suffix(name, ".beta") ||
        has_suffix(name, ".running_mean") ||
        has_suffix(name, ".running_var")) {
        return engine::assets::TensorStorageType::F32;
    }
    return requested_storage_type;
}

}  // namespace

void load_seed_vc_tensor_store(
    SeedVcTensorStore & weights,
    std::shared_ptr<const engine::assets::TensorSource> source,
    const std::string & name,
    engine::core::BackendConfig backend,
    const std::vector<std::string> & generated_constants,
    const std::vector<std::string> & generated_constant_suffixes,
    engine::assets::TensorStorageType storage_type) {
    if (source == nullptr) {
        throw std::runtime_error("Seed-VC " + name + " weights are missing");
    }
    weights.source_path = source->source_path();
    weights.name = name;
    weights.execution_context = std::make_shared<engine::core::ExecutionContext>(backend);
    weights.store = std::make_shared<engine::core::BackendWeightStore>(
        weights.execution_context->backend(),
        weights.execution_context->backend_type(),
        "seed_vc." + name + ".weights",
        256ull * 1024ull * 1024ull);

    const auto tensors = source->tensors();
    weights.tensors.reserve(tensors.size());
    for (const auto & tensor : tensors) {
        if (contains_name(generated_constants, tensor.name) ||
            contains_suffix(generated_constant_suffixes, tensor.name)) {
            ++weights.skipped_generated_constant_count;
            continue;
        }
        if (!is_float_dtype(tensor.dtype)) {
            throw std::runtime_error(
                "Seed-VC " + name + " contains unsupported non-floating tensor: " + tensor.name);
        }
        weights.parameter_count += tensor_elements(tensor.shape, tensor.name);
        weights.tensors.emplace(
            tensor.name,
            weights.store->load_tensor(
                *source,
                tensor.name,
                storage_type_for_seed_vc_tensor(tensor.name, tensor.shape, storage_type),
                tensor.shape));
        ++weights.loaded_tensor_count;
    }
    if (weights.loaded_tensor_count <= 0) {
        throw std::runtime_error("Seed-VC " + name + " safetensors file contains no loadable tensors");
    }
    weights.store->upload();
    source->release_storage();
}

}  // namespace engine::models::seed_vc
