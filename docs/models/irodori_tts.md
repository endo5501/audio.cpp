# Irodori-TTS

Irodori-TTS is Japanese TTS under `--family irodori_tts`. It supports no-reference speech, optional reference-audio voice cloning, and caption-based voice design.

The default downloadable package is the GGUF v4 Small Q8_0 checkpoint. v4 Small is the preferred path for new use because one checkpoint covers no-reference TTS, voice cloning, and voice design. The older v3 packages remain supported for existing users and local validation.

## Variants

| Variant | Package | Tasks | Notes |
|---|---|---|---|
| v4 Small | `Irodori-TTS-v4-Small-GGUF` | `tts`, `clon`, `vdes` | Bundles its v4 tokenizer and supports caption conditioning in the same checkpoint. |
| 500M v3 | `Irodori-TTS-500M-v3-GGUF` | `tts`, `clon` | Uses the shared llm-jp tokenizer layout in the original safetensors package. |
| 600M v3 VoiceDesign | `Irodori-TTS-600M-v3-VoiceDesign-GGUF` | `tts`, `clon`, `vdes` | Adds caption conditioning for voice design. |

v4 GGUF packages are published in both `q8_0` and `f16`. v3 GGUF packages are also available in `q8_0` and `f16`.

> **v4 reference-conditioning note:** v4 voice-clone or reference+caption generations may add a short extra phrase near the end of the clip — with a reference *and* a caption together, in 8-9 runs out of 10 rather than occasionally. This behavior is also reproducible in the upstream Python path, so it is a v4 model behaviour rather than a GGUF-only issue. The cause is the duration predictor asking for more time than the text needs; set `duration_correction=true` to bound the length instead (see [Duration correction](#duration-correction)). A different seed does not help — the predicted length is deterministic.

## Quick Start

No-reference v4 speech:

```bash
audiocpp_cli --task tts --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "今日は短い確認です。やさしく、聞き取りやすい声でお願いします。" \
  --request-option no_ref=true \
  --out out.wav
```

v4 voice cloning:

```bash
audiocpp_cli --task clon --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。" \
  --voice-ref models/Irodori-TTS-v4-Small/samples/clone_ref1.wav \
  --out out.wav
```

v4 voice design:

```bash
audiocpp_cli --task vdes --family irodori_tts \
  --model models/Irodori-TTS-v4-Small-GGUF/irodori-tts-v4-small-q8_0.gguf \
  --backend cuda --language ja \
  --text "本日はお越しいただき、誠にありがとうございます。" \
  --request-option caption="落ち着いた大人の男性。深く響く声で丁寧に話している。" \
  --request-option no_ref=true \
  --request-option guidance_scale=3 \
  --out out.wav
```

v3 voice cloning:

```bash
audiocpp_cli --task clon --family irodori_tts \
  --model models/Irodori-TTS-500M-v3-GGUF/irodori-tts-500m-v3-q8_0.gguf \
  --backend cuda --language ja \
  --text "同じ声で短く話します。" \
  --voice-ref models/Irodori-TTS-500M-v3/samples/clone_ref1.wav \
  --request-option no_ref=false \
  --out out.wav
```

## Request Options

v4 uses the normalized schema-v1 option names directly. New requests should use these names:

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `language` | `ja` | `ja` | Text language code; Irodori-TTS is Japanese-only. |
| `caption` | text | empty | Voice-design caption; only useful on caption-conditioned checkpoints. |
| `no_ref` | bool | `true` unless a reference is provided | Use no-reference generation. Set `false` with `--voice-ref` for reference conditioning. |
| `num_inference_steps` | integer | `40` | RF diffusion steps. |
| `duration_sec` | seconds | unset | Explicit output duration; omitted uses predicted duration. |
| `duration_scale` | float | `1.0` | Multiplier for predicted duration. |
| `min_duration_sec` | seconds | `0.5` | Minimum generated duration. |
| `max_duration_sec` | seconds | `30` | Maximum generated duration. |
| `text_chunk_mode` | `japanese`, `endline` | `endline` | Long-form chunking mode. |
| `text_chunk_size` | integer | model text window | Maximum characters per outer chunk. |
| `text_guidance_scale` | float | `3.0` | Text CFG strength. |
| `speaker_guidance_scale` | float | `5.0` | Speaker CFG strength. |
| `caption_guidance_scale` | float | `3.0` | Caption CFG strength. |
| `guidance_scale` | float | unset | Override all CFG strengths when set. |
| `guidance_mode` | `independent`, `joint`, `alternating` | `independent` | CFG combination mode. |
| `guidance_min_t` | float | `0.5` | Minimum diffusion timestep where guidance is active. |
| `guidance_max_t` | float | `1.0` | Maximum diffusion timestep where guidance is active. |
| `seed` | integer | random | Generation seed. |
| `trim_tail` | bool | `true` | Trim trailing silence-like samples. |
| `duration_correction` | bool | `false` | Bound the generated length to roughly what the text needs. See below. |
| `duration_correction_rate` | float | `0.207` | Seconds per spoken codepoint for the character rule. Japanese calibration. |
| `duration_correction_margin_sec` | seconds | `0.4` | Trailing room left after the text ends by the character rule. |

### Duration correction

With a reference voice **and** a caption, the v4 duration predictor asks for more
time than the text needs, and the model fills the surplus with a phrase that is
not in the input — measured at 8-9 runs in 10, and reproducible in the upstream
PyTorch path, so it is a model behaviour rather than a GGUF-only issue.
The surplus is what drives it: at 0.5 s or less it does not occur, at 1.9 s or
more it almost always does.

`duration_correction=true` bounds the target length to the smallest of

1. the normal prediction,
2. the prediction re-run with the speaker and caption conditions disabled, and
3. `codepoints(text, excluding punctuation) * duration_correction_rate + duration_correction_margin_sec`.

Both extra terms are needed. The no-condition prediction is accurate for a
medium-length sentence but over-predicts a short one by +83%, where the
character rule takes over; the character rule alone leaves enough room for a
residue when the caption asks for fast speech, where the no-condition
prediction takes over.

The correction only shortens, is applied per chunk, and is ignored when
`duration_sec` is set. `min_duration_sec` / `max_duration_sec` still clamp the
result. Its cost is one extra condition-encoder pass per chunk, reported as
`irodori_tts.noref_condition_ms`.

Shortening below the no-condition prediction makes the artifact come back, so
the rule is not "shorter is safer" — the terms above are the calibrated bound.

## Session Options

| Option | Values | Default | Meaning |
|---|---|---:|---|
| `irodori_tts.weight_type` | `native`, `f32`, `f16`, `bf16`, `q8_0` | `native` | Model weight storage type. |
| `irodori_tts.codec_weight_type` | `native`, `f32`, `f16`, `q8_0` | `native` | DACVAE codec weight storage type. |
| `irodori_tts.mem_saver` | bool | `true` | Release staged runtime graphs after request phases. |
| `irodori_tts.reference_cache_slots` | integer | `1` | Prepared reference-speaker cache slots; use `0` to disable reuse. |
| `irodori_tts.condition_graph_arena_mb` | MiB | `256` | Condition encoder graph arena size. |
| `irodori_tts.rf_graph_arena_mb` | MiB | `768` | RF sampler graph arena size. |
| `irodori_tts.codec_graph_arena_mb` | MiB | `512` | DACVAE codec graph arena size. |
| `irodori_tts.condition_weight_context_mb` | MiB | `32` | Condition encoder weight metadata context size. |
| `irodori_tts.rf_weight_context_mb` | MiB | `32` | RF sampler weight metadata context size. |
| `irodori_tts.codec_weight_context_mb` | MiB | `32` | DACVAE codec weight metadata context size. |

## Compatibility

The v3 runtime accepts the old option names below for existing local scripts and older standalone GGUF packages. Prefer the v1 names for new requests.

| Legacy option | v1 option |
|---|---|
| `duration_seconds` | `duration_sec` |
| `min_seconds` | `min_duration_sec` |
| `max_seconds` | `max_duration_sec` |
| `mem_saver` | `irodori_tts.mem_saver` |
| `reference_cache_slots` | `irodori_tts.reference_cache_slots` |
