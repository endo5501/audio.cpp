#include "audiocpp_c_api.h"

#include "engine/framework/audio/wav_reader.h"
#include "engine/framework/core/module.h"
#include "engine/framework/runtime/model.h"
#include "engine/framework/runtime/registry.h"
#include "engine/framework/runtime/session.h"
#include "engine/models/irodori_tts/session.h"

#include <atomic>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <dlfcn.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// Abort flag holder whose lifetime is independent of any synthesis context.
// Created once per TTS session and outlives context init/free, so an abort
// issued after the context is freed never writes to freed memory (F111 lesson).
//
// ABI CONTRACT: this struct MUST remain a single std::atomic<bool> and stay
// byte-compatible with qwen3_tts_abort_handle in the qwen3-tts.cpp fork.
// NovelViewer shares one session abort handle across both engine DLLs (a
// handle allocated by either library may be wired into the other's init), so
// any layout change here silently breaks cross-engine abort. Change both
// forks in lockstep or not at all.
struct audiocpp_abort_handle {
  std::atomic<bool> abort_flag{false};
};

struct audiocpp_ctx {
  engine::runtime::ModelRegistry registry;
  std::unique_ptr<engine::runtime::ILoadedVoiceModel> model;
  std::unique_ptr<engine::runtime::IVoiceTaskSession> session;
  engine::runtime::IOfflineVoiceTaskSession *offline = nullptr;
  engine::models::irodori_tts::IrodoriTTSSession *irodori = nullptr;
  std::vector<float> audio;
  int sample_rate = 0;
  std::string last_error;
  bool loaded = false;

  // Single-entry cache for the decoded reference WAV. audiocpp_synthesize is
  // called repeatedly with the same ref_wav_path during a clone session, so we
  // avoid re-reading and re-decoding the file every call. The cache is keyed on
  // the resolved path plus the file's size and last-write time; a change to any
  // of them invalidates the entry. If the file cannot be stat'd we fall back to
  // an uncached read and leave the cache invalid.
  std::string ref_cache_path;
  std::uintmax_t ref_cache_size = 0;
  std::filesystem::file_time_type ref_cache_mtime{};
  engine::runtime::AudioBuffer ref_cache_buffer;
  bool ref_cache_valid = false;
};

namespace {

std::filesystem::path path_from_utf8(const char *utf8) {
  if (utf8 == nullptr) {
    return {};
  }
#ifdef _WIN32
  const std::string narrow(utf8);
  if (narrow.empty()) {
    return {};
  }
  const int len = MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, nullptr, 0);
  if (len <= 0) {
    return std::filesystem::path(narrow);
  }
  std::wstring wide(static_cast<size_t>(len - 1), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, wide.data(), len);
  return std::filesystem::path(wide);
#else
  return std::filesystem::path(std::string(utf8));
#endif
}

// Directory containing this shared library, used to locate the bundled model
// spec next to the DLL/dylib (independent of the executable location).
std::filesystem::path this_module_dir() {
#ifdef _WIN32
  HMODULE module = nullptr;
  if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCWSTR>(&audiocpp_init),
                         &module) &&
      module != nullptr) {
    std::wstring buffer(1024, L'\0');
    for (;;) {
      const DWORD written = GetModuleFileNameW(
          module, buffer.data(), static_cast<DWORD>(buffer.size()));
      if (written == 0) {
        break;
      }
      if (written < buffer.size()) {
        buffer.resize(written);
        return std::filesystem::path(buffer).parent_path();
      }
      buffer.resize(buffer.size() * 2);
    }
  }
  return {};
#elif defined(__APPLE__)
  Dl_info info;
  if (dladdr(reinterpret_cast<const void *>(&audiocpp_init), &info) != 0 &&
      info.dli_fname != nullptr) {
    return std::filesystem::path(info.dli_fname).parent_path();
  }
  return {};
#else
  return {};
#endif
}

// Resolve the Irodori model spec: prefer a model_specs/irodori_tts.json bundled
// next to the shared library, else fall back to <model_dir>/irodori_tts.json.
std::optional<std::filesystem::path>
resolve_model_spec(const std::filesystem::path &model_dir) {
  const std::filesystem::path module_dir = this_module_dir();
  std::error_code ec;
  if (!module_dir.empty()) {
    const std::filesystem::path bundled =
        module_dir / "model_specs" / "irodori_tts.json";
    if (std::filesystem::exists(bundled, ec)) {
      return bundled;
    }
  }
  const std::filesystem::path fallback = model_dir / "irodori_tts.json";
  if (std::filesystem::exists(fallback, ec)) {
    return fallback;
  }
  return std::nullopt;
}

engine::runtime::AudioBuffer read_audio_buffer(const std::filesystem::path &path) {
  const auto wav = engine::audio::read_wav_f32(path);
  return engine::runtime::AudioBuffer{wav.sample_rate, wav.channels, wav.samples};
}

// Returns the decoded reference audio for `path`, reusing ctx's single-entry
// cache when the file's identity (key path + size + last-write time) is
// unchanged. On any stat failure the cache is bypassed and invalidated and the
// file is read directly. The returned reference is owned by ctx and remains
// valid until the next call; callers copy it into the request.
const engine::runtime::AudioBuffer &
get_ref_audio_buffer(audiocpp_ctx &ctx, const std::string &key,
                     const std::filesystem::path &path) {
  std::error_code size_ec;
  const std::uintmax_t size = std::filesystem::file_size(path, size_ec);
  std::error_code mtime_ec;
  const std::filesystem::file_time_type mtime =
      std::filesystem::last_write_time(path, mtime_ec);
  if (!size_ec && !mtime_ec) {
    if (ctx.ref_cache_valid && ctx.ref_cache_path == key &&
        ctx.ref_cache_size == size && ctx.ref_cache_mtime == mtime) {
      return ctx.ref_cache_buffer;
    }
    ctx.ref_cache_buffer = read_audio_buffer(path);
    ctx.ref_cache_path = key;
    ctx.ref_cache_size = size;
    ctx.ref_cache_mtime = mtime;
    ctx.ref_cache_valid = true;
    return ctx.ref_cache_buffer;
  }
  // Cannot stat the file: bypass the cache, invalidate any stale entry, and
  // read directly (reusing the member as scratch storage).
  ctx.ref_cache_valid = false;
  ctx.ref_cache_buffer = read_audio_buffer(path);
  return ctx.ref_cache_buffer;
}

// Create the offline task session, preferring the platform GPU backend
// (Vulkan on Windows, Metal on macOS) and falling back to CPU on failure.
std::unique_ptr<engine::runtime::IVoiceTaskSession>
create_session_with_fallback(const engine::runtime::ILoadedVoiceModel &model,
                             int threads) {
  const engine::runtime::TaskSpec task_spec{
      engine::runtime::VoiceTaskKind::Tts,
      engine::runtime::RunMode::Offline,
  };
#if defined(__APPLE__)
  const engine::core::BackendType preferred = engine::core::BackendType::Metal;
#else
  const engine::core::BackendType preferred = engine::core::BackendType::Vulkan;
#endif
  try {
    engine::runtime::SessionOptions options;
    options.backend.type = preferred;
    options.backend.threads = threads;
    return model.create_task_session(task_spec, options);
  } catch (const std::exception &) {
    // GPU backend unavailable or failed to initialize; fall back to CPU.
  }
  engine::runtime::SessionOptions options;
  options.backend.type = engine::core::BackendType::Cpu;
  options.backend.threads = threads;
  return model.create_task_session(task_spec, options);
}

}  // namespace

audiocpp_abort_handle *audiocpp_create_abort_handle(void) {
  return new (std::nothrow) audiocpp_abort_handle();
}

void audiocpp_free_abort_handle(audiocpp_abort_handle *handle) { delete handle; }

void audiocpp_abort(audiocpp_abort_handle *handle) {
  if (handle == nullptr) {
    return;
  }
  handle->abort_flag.store(true, std::memory_order_release);
}

void audiocpp_reset_abort(audiocpp_abort_handle *handle) {
  if (handle == nullptr) {
    return;
  }
  handle->abort_flag.store(false, std::memory_order_release);
}

audiocpp_ctx *audiocpp_init(const char *model_dir, int n_threads,
                            audiocpp_abort_handle *abort_handle) {
  if (model_dir == nullptr) {
    return nullptr;
  }
  auto *ctx = new (std::nothrow) audiocpp_ctx();
  if (ctx == nullptr) {
    return nullptr;
  }
  try {
    const std::filesystem::path model_path = path_from_utf8(model_dir);

    const int threads = n_threads > 0 ? n_threads : 4;
#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    ctx->registry = engine::runtime::make_default_registry();

    engine::runtime::ModelLoadRequest load_request;
    load_request.model_path = model_path;
    load_request.family_hint = "irodori_tts";
    if (const auto spec = resolve_model_spec(model_path)) {
      load_request.model_spec_override = *spec;
    }
    ctx->model = ctx->registry.load(load_request);
    if (ctx->model == nullptr) {
      delete ctx;
      return nullptr;
    }

    ctx->session = create_session_with_fallback(*ctx->model, threads);
    ctx->offline = dynamic_cast<engine::runtime::IOfflineVoiceTaskSession *>(
        ctx->session.get());
    if (ctx->offline == nullptr) {
      delete ctx;
      return nullptr;
    }
    ctx->irodori =
        dynamic_cast<engine::models::irodori_tts::IrodoriTTSSession *>(
            ctx->session.get());
    if (ctx->irodori != nullptr && abort_handle != nullptr) {
      ctx->irodori->set_abort_flag(&abort_handle->abort_flag);
    }
    ctx->session->prepare(engine::runtime::SessionPreparationRequest{});
    ctx->loaded = true;
    return ctx;
  } catch (const std::exception &) {
    // Invalid model path / load failure: return null without crashing.
    delete ctx;
    return nullptr;
  } catch (...) {
    delete ctx;
    return nullptr;
  }
}

int audiocpp_is_loaded(const audiocpp_ctx *ctx) {
  if (ctx == nullptr) {
    return 0;
  }
  return ctx->loaded ? 1 : 0;
}

void audiocpp_free(audiocpp_ctx *ctx) {
  // Note: the abort handle is intentionally NOT freed here; it is owned by the
  // caller and may outlive the context.
  delete ctx;
}

int audiocpp_synthesize(audiocpp_ctx *ctx, const char *text,
                        const char *ref_wav_path, const char *caption,
                        float speaker_guidance_scale,
                        float caption_guidance_scale, int num_inference_steps) {
  if (ctx == nullptr) {
    return -1;
  }
  if (ctx->offline == nullptr || text == nullptr) {
    ctx->last_error = "Irodori-TTS context is not ready or text is null";
    return -1;
  }
  try {
    engine::runtime::TaskRequest request;
    request.text_input = engine::runtime::Transcript{std::string(text), "ja"};

    if (caption != nullptr && caption[0] != '\0') {
      request.options["caption"] = std::string(caption);
    }
    // A scale of exactly 0.0 is semantically meaningful (it disables CFG for
    // that stream in the engine), so any non-negative value is forwarded.
    // Only a negative value means "use the engine default".
    if (speaker_guidance_scale >= 0.0F) {
      request.options["speaker_guidance_scale"] =
          std::to_string(speaker_guidance_scale);
    }
    if (caption_guidance_scale >= 0.0F) {
      request.options["caption_guidance_scale"] =
          std::to_string(caption_guidance_scale);
    }
    if (num_inference_steps > 0) {
      request.options["num_inference_steps"] =
          std::to_string(num_inference_steps);
    }
    if (ref_wav_path != nullptr && ref_wav_path[0] != '\0') {
      const std::string ref_key(ref_wav_path);
      request.voice = engine::runtime::VoiceCondition{};
      request.voice->speaker = engine::runtime::VoiceReference{};
      request.voice->speaker->audio =
          get_ref_audio_buffer(*ctx, ref_key, path_from_utf8(ref_wav_path));
    }

    engine::runtime::TaskResult result = ctx->offline->run(request);
    if (!result.audio_output.has_value()) {
      ctx->last_error = "Irodori-TTS produced no audio output";
      return -1;
    }
    ctx->audio = std::move(result.audio_output->samples);
    ctx->sample_rate = result.audio_output->sample_rate;
    ctx->last_error.clear();
    return 0;
  } catch (const std::exception &ex) {
    ctx->last_error = ex.what();
    return -1;
  } catch (...) {
    ctx->last_error = "unknown Irodori-TTS synthesis error";
    return -1;
  }
}

const float *audiocpp_get_audio(const audiocpp_ctx *ctx) {
  if (ctx == nullptr || ctx->audio.empty()) {
    return nullptr;
  }
  return ctx->audio.data();
}

int audiocpp_get_audio_length(const audiocpp_ctx *ctx) {
  if (ctx == nullptr) {
    return 0;
  }
  return static_cast<int>(ctx->audio.size());
}

int audiocpp_get_sample_rate(const audiocpp_ctx *ctx) {
  if (ctx == nullptr) {
    return 0;
  }
  return ctx->sample_rate;
}

const char *audiocpp_get_error(const audiocpp_ctx *ctx) {
  if (ctx == nullptr) {
    return "null context";
  }
  return ctx->last_error.c_str();
}
