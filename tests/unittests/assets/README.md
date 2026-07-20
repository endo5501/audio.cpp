# Test Assets

Everything under this directory is checked into the repo so unit tests can run
without downloaded models or user-local paths.

Current fixtures:

- `registry/silero_vad_registry.txt`
  - config-driven loader enablement for the Silero VAD trace parity test
- `tokenizers/tokenizer-1.model`
  - real SentencePiece model used by the framework tokenizer unit test
- `framework/audio/tone_440hz_16k_mono.mp3`
  - 0.2s 440Hz tone, 16kHz mono, 32kbps, no ID3 tag (864 bytes). Real encoder
    output for the audio reader unit test; WAV inputs are generated at runtime
    but MP3 needs an encoder, so this one is checked in.
