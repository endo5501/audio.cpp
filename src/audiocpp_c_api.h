#ifndef AUDIOCPP_C_API_H
#define AUDIOCPP_C_API_H

// C ABI shim over the audio.cpp engine runtime for NovelViewer's Dart FFI
// integration. Exposes the Irodori-TTS voice-cloning + caption model as a
// single-function synthesis surface with an abort handle whose lifetime is
// independent of any synthesis context (mirrors qwen3_tts_c_api; the
// independent handle lifetime fixes the F111 use-after-free class of bug).

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  define AUDIOCPP_API __declspec(dllexport)
#else
#  define AUDIOCPP_API __attribute__((visibility("default")))
#endif

typedef struct audiocpp_ctx audiocpp_ctx;

// Abort handle: holds the atomic abort flag with a lifetime that is independent
// of any synthesis context. Created once (per TTS session) and outlives context
// init/free, so aborting after the context is freed never touches freed memory.
typedef struct audiocpp_abort_handle audiocpp_abort_handle;

// Abort handle lifecycle.
AUDIOCPP_API audiocpp_abort_handle *audiocpp_create_abort_handle(void);
AUDIOCPP_API void audiocpp_free_abort_handle(audiocpp_abort_handle *handle);

// Abort (thread-safe, callable from any thread). Operates on the abort handle,
// never on the context, so it stays safe after audiocpp_free().
AUDIOCPP_API void audiocpp_abort(audiocpp_abort_handle *handle);
AUDIOCPP_API void audiocpp_reset_abort(audiocpp_abort_handle *handle);

// Lifecycle.
// `model_dir` is the Irodori-TTS-600M-v3-VoiceDesign model root (its
// llm-jp-3-150m and Semantic-DACVAE-Japanese-32dim siblings are resolved
// relative to it via the model spec). `abort_handle` may be null; when provided,
// its flag is checked during synthesis. The handle is owned by the caller and is
// NOT freed by audiocpp_free. Returns null on failure.
AUDIOCPP_API audiocpp_ctx *audiocpp_init(const char *model_dir, int n_threads,
                                         audiocpp_abort_handle *abort_handle);
AUDIOCPP_API int audiocpp_is_loaded(const audiocpp_ctx *ctx);
AUDIOCPP_API void audiocpp_free(audiocpp_ctx *ctx);

// Unified synthesis. `ref_wav_path` and `caption` are both nullable; the NULL
// combinations give four modes with one function:
//   ref=NULL,   caption=NULL   -> plain TTS
//   ref=path,   caption=NULL   -> voice clone
//   ref=NULL,   caption=text   -> caption / voice design
//   ref=path,   caption=text   -> clone + caption combined
// An empty caption string is treated as no caption. `speaker_guidance_scale`,
// `caption_guidance_scale`, and `num_inference_steps` fall back to engine
// defaults (5.0 / 3.0 / 40) when <= 0. Returns 0 on success, non-zero on error
// (including abort); on error audiocpp_get_error(ctx) returns the message.
AUDIOCPP_API int audiocpp_synthesize(audiocpp_ctx *ctx, const char *text,
                                     const char *ref_wav_path,
                                     const char *caption,
                                     float speaker_guidance_scale,
                                     float caption_guidance_scale,
                                     int num_inference_steps);

// Result access (valid until the next synthesize call or audiocpp_free).
AUDIOCPP_API const float *audiocpp_get_audio(const audiocpp_ctx *ctx);
AUDIOCPP_API int audiocpp_get_audio_length(const audiocpp_ctx *ctx);
AUDIOCPP_API int audiocpp_get_sample_rate(const audiocpp_ctx *ctx);
AUDIOCPP_API const char *audiocpp_get_error(const audiocpp_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif  // AUDIOCPP_C_API_H
