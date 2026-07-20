#include "engine/framework/audio/wav_reader.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename T>
void write_le(std::ofstream & output, T value) {
    output.write(reinterpret_cast<const char *>(&value), sizeof(T));
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_bytes(std::ofstream & output, const char * bytes, std::streamsize count) {
    output.write(bytes, count);
    if (!output) {
        throw std::runtime_error("failed to write test WAV");
    }
}

void write_pcm24_sample(std::ofstream & output, int32_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) & 0x00FFFFFFu;
    const char bytes[3] = {
        static_cast<char>(bits & 0xFFu),
        static_cast<char>((bits >> 8) & 0xFFu),
        static_cast<char>((bits >> 16) & 0xFFu),
    };
    write_bytes(output, bytes, 3);
}

void write_pcm24_wav(
    const std::filesystem::path & path,
    int sample_rate,
    int channels,
    const std::vector<int32_t> & samples) {
    const uint32_t data_bytes = static_cast<uint32_t>(samples.size() * 3);
    const uint16_t block_align = static_cast<uint16_t>(channels * 3);
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate * block_align);

    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test WAV: " + path.string());
    }
    write_bytes(output, "RIFF", 4);
    write_le<uint32_t>(output, 36u + data_bytes);
    write_bytes(output, "WAVE", 4);
    write_bytes(output, "fmt ", 4);
    write_le<uint32_t>(output, 16u);
    write_le<uint16_t>(output, 1u);
    write_le<uint16_t>(output, static_cast<uint16_t>(channels));
    write_le<uint32_t>(output, static_cast<uint32_t>(sample_rate));
    write_le<uint32_t>(output, byte_rate);
    write_le<uint16_t>(output, block_align);
    write_le<uint16_t>(output, 24u);
    write_bytes(output, "data", 4);
    write_le<uint32_t>(output, data_bytes);
    for (const int32_t sample : samples) {
        write_pcm24_sample(output, sample);
    }
}

void require_near(float actual, float expected, const std::string & label) {
    if (std::fabs(actual - expected) > 1.0e-7F) {
        throw std::runtime_error(label + " mismatch");
    }
}

// ---------------------------------------------------------------------------
// In-memory WAV builders.
//
// These tests exercise the byte layouts that real recorders emit, so the
// fixtures are assembled here rather than checked in as binaries: what each
// case is probing stays readable next to the assertion.
// ---------------------------------------------------------------------------

template <typename T>
void append_le(std::string & out, T value) {
    char bytes[sizeof(T)];
    for (size_t i = 0; i < sizeof(T); ++i) {
        bytes[i] = static_cast<char>((static_cast<uint64_t>(value) >> (8 * i)) & 0xFFu);
    }
    out.append(bytes, sizeof(T));
}

// A chunk whose declared size deliberately may differ from the payload it
// actually carries, so the "header lies about the size" cases are expressible.
std::string chunk_with_size(const std::string & id, uint32_t declared_size, const std::string & payload) {
    std::string out = id;
    append_le<uint32_t>(out, declared_size);
    out += payload;
    return out;
}

std::string chunk(const std::string & id, const std::string & payload) {
    std::string out = chunk_with_size(id, static_cast<uint32_t>(payload.size()), payload);
    if (payload.size() % 2 == 1) {
        out.push_back('\0');
    }
    return out;
}

std::string build_wav(const std::vector<std::string> & chunks, const std::string & trailing = std::string()) {
    std::string body;
    for (const std::string & c : chunks) {
        body += c;
    }
    body += trailing;

    std::string out = "RIFF";
    append_le<uint32_t>(out, static_cast<uint32_t>(4 + body.size()));
    out += "WAVE";
    out += body;
    return out;
}

// 16-byte PCM-style fmt body.
std::string fmt_body(uint16_t format_tag, uint16_t channels, uint32_t sample_rate, uint16_t bits) {
    const uint16_t block_align = static_cast<uint16_t>(channels * (bits / 8));
    std::string p;
    append_le<uint16_t>(p, format_tag);
    append_le<uint16_t>(p, channels);
    append_le<uint32_t>(p, sample_rate);
    append_le<uint32_t>(p, sample_rate * block_align);
    append_le<uint16_t>(p, block_align);
    append_le<uint16_t>(p, bits);
    return p;
}

// The fixed 14-byte tail shared by every KSDATAFORMAT_SUBTYPE_* GUID; only the
// leading 2 bytes vary and carry the real format tag.
const unsigned char kGuidTail[14] = {
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00,
    0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71,
};

// 40-byte WAVE_FORMAT_EXTENSIBLE fmt body, as macOS AVAudioRecorder writes it.
std::string fmt_body_extensible(uint16_t sub_format_tag, uint16_t channels, uint32_t sample_rate, uint16_t bits) {
    std::string p = fmt_body(0xFFFEu, channels, sample_rate, bits);
    append_le<uint16_t>(p, 22u);   // cbSize
    append_le<uint16_t>(p, bits);  // wValidBitsPerSample
    append_le<uint32_t>(p, 0u);    // dwChannelMask
    append_le<uint16_t>(p, sub_format_tag);
    p.append(reinterpret_cast<const char *>(kGuidTail), sizeof(kGuidTail));
    return p;
}

// 18-byte fmt body that declares EXTENSIBLE but stops before the SubFormat
// GUID: record_windows copies only sizeof(WAVEFORMATEX), so this shape exists.
std::string fmt_body_extensible_truncated(uint16_t channels, uint32_t sample_rate, uint16_t bits) {
    std::string p = fmt_body(0xFFFEu, channels, sample_rate, bits);
    append_le<uint16_t>(p, 22u);  // cbSize claims an extension that is not there
    return p;
}

std::string pcm16_payload(const std::vector<int16_t> & samples) {
    std::string p;
    for (const int16_t sample : samples) {
        append_le<uint16_t>(p, static_cast<uint16_t>(sample));
    }
    return p;
}

std::string float32_payload(const std::vector<float> & samples) {
    std::string p;
    for (const float sample : samples) {
        uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(sample), "float32 must be 4 bytes");
        std::memcpy(&bits, &sample, sizeof(bits));
        append_le<uint32_t>(p, bits);
    }
    return p;
}

void require_throws(const std::string & input, const std::string & label) {
    bool threw = false;
    try {
        (void)engine::audio::read_wav_f32(std::string_view(input));
    } catch (const std::exception &) {
        threw = true;
    }
    require(threw, label + " should have failed");
}

const std::vector<int16_t> kPcm16Samples = {0, 32767, -32768, -1};

void require_pcm16_samples(const engine::audio::WavData & wav, const std::string & label) {
    require(wav.samples.size() == 4, label + " sample count mismatch");
    require_near(wav.samples[0], 0.0F, label + " zero");
    require_near(wav.samples[1], 32767.0F / 32768.0F, label + " max positive");
    require_near(wav.samples[2], -1.0F, label + " min negative");
    require_near(wav.samples[3], -1.0F / 32768.0F, label + " negative one");
}

// --- 1.2 EXTENSIBLE carrying a PCM SubFormat -------------------------------
void test_extensible_pcm16() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body_extensible(0x0001u, 1, 16000, 16)),
        chunk("data", pcm16_payload(kPcm16Samples)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "extensible PCM16 sample rate mismatch");
    require(wav.channels == 1, "extensible PCM16 channel count mismatch");
    require_pcm16_samples(wav, "extensible PCM16");
}

// --- 1.3 EXTENSIBLE carrying an IEEE float SubFormat -----------------------
void test_extensible_float32() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body_extensible(0x0003u, 2, 48000, 32)),
        chunk("data", float32_payload({0.0F, 0.5F, -0.25F, 1.0F})),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 48000, "extensible float32 sample rate mismatch");
    require(wav.channels == 2, "extensible float32 channel count mismatch");
    require(wav.samples.size() == 4, "extensible float32 sample count mismatch");
    require_near(wav.samples[0], 0.0F, "extensible float32 zero");
    require_near(wav.samples[1], 0.5F, "extensible float32 half");
    require_near(wav.samples[2], -0.25F, "extensible float32 negative quarter");
    require_near(wav.samples[3], 1.0F, "extensible float32 one");
}

// --- 1.4 EXTENSIBLE without the GUID falls back to the bit depth -----------
void test_extensible_truncated_fmt() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body_extensible_truncated(1, 16000, 16)),
        chunk("data", pcm16_payload(kPcm16Samples)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "truncated extensible sample rate mismatch");
    require(wav.channels == 1, "truncated extensible channel count mismatch");
    require_pcm16_samples(wav, "truncated extensible");
}

// --- 1.5 EXTENSIBLE with a SubFormat we cannot decode ----------------------
void test_extensible_unknown_subformat() {
    // 0x0006 is A-law: a real format tag, but not one this reader decodes.
    const std::string input = build_wav({
        chunk("fmt ", fmt_body_extensible(0x0006u, 1, 16000, 16)),
        chunk("data", pcm16_payload(kPcm16Samples)),
    });

    require_throws(input, "extensible with unknown SubFormat");
}

// 32 bits is ambiguous without the GUID: integer PCM32 and IEEE float32 share
// the layout, and guessing float would reinterpret integer samples as floats -
// silent garbage instead of an honest failure.
void test_extensible_truncated_fmt_32bit_is_rejected() {
    std::string payload;
    for (int i = 0; i < 4; ++i) {
        append_le<uint32_t>(payload, static_cast<uint32_t>(1000 * (i + 1)));
    }
    const std::string input = build_wav({
        chunk("fmt ", fmt_body_extensible_truncated(1, 48000, 32)),
        chunk("data", payload),
    });

    require_throws(input, "truncated extensible at 32 bits");
}

// Debris that happens to spell "data" must not truncate audio already read.
void test_later_data_chunk_does_not_replace_the_first() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk("data", pcm16_payload(kPcm16Samples)),
        chunk("data", pcm16_payload({7})),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "first data chunk wins");
}

void test_later_fmt_chunk_does_not_replace_the_first() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk("data", pcm16_payload(kPcm16Samples)),
        chunk("fmt ", fmt_body(1u, 2, 44100, 16)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "first fmt sample rate must win");
    require(wav.channels == 1, "first fmt channel count must win");
    require_pcm16_samples(wav, "first fmt wins");
}

// --- 1.6 Padding chunks around fmt -----------------------------------------
void test_junk_and_filler_chunks() {
    const std::string input = build_wav({
        chunk("JUNK", std::string(28, '\0')),
        chunk("fmt ", fmt_body_extensible(0x0001u, 1, 16000, 16)),
        chunk("FLLR", std::string(3984, '\0')),
        chunk("data", pcm16_payload(kPcm16Samples)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "JUNK/FLLR sample rate mismatch");
    require_pcm16_samples(wav, "JUNK/FLLR");
}

// --- 1.7 Debris after a complete parse -------------------------------------
void test_trailing_garbage_after_data() {
    // record_windows leaves its sink writer's old header behind; the reader
    // walks into it and reads a bogus (non-printable) id. That id is the
    // debris signal: the walk stops, and the well-sized data payload survives
    // the trailing-chunk scan untouched.
    std::string trailing;
    // Embedded NULs, so the length must be explicit: appending this as a C
    // string would stop at the first byte and silently shrink the fixture.
    trailing.append("\x01\x00\x00\x00", 4); // garbage chunk id
    append_le<uint32_t>(trailing, 131072);  // size far beyond the input
    trailing.append(28, '\0');

    const std::string input = build_wav(
        {
            chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
            chunk("data", pcm16_payload(kPcm16Samples)),
        },
        trailing);

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "trailing garbage sample rate mismatch");
    require_pcm16_samples(wav, "trailing garbage");
}

// The walk dies mid-header after fmt and data are complete. The clamp cannot
// defuse this one (the header itself is cut short), so it is the case that
// actually exercises the completed-parse tolerance.
void test_truncated_chunk_header_after_data() {
    const std::string input = build_wav(
        {
            chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
            chunk("data", pcm16_payload(kPcm16Samples)),
        },
        std::string("LIST\x01\x01", 6));

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "truncated trailing header");
}

// --- 1.8 data chunk that claims more than the input holds -------------------
void test_oversized_data_chunk_size() {
    const std::string payload = pcm16_payload(kPcm16Samples);
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk_with_size("data", static_cast<uint32_t>(payload.size() + 100), payload),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(wav.sample_rate == 16000, "oversized data sample rate mismatch");
    require_pcm16_samples(wav, "oversized data");
}

// A streaming writer that never patched the header: data claims 0xFFFFFFFF and
// the payload genuinely runs to EOF. Everything up to EOF is audio.
void test_unpatched_streaming_size_reads_to_eof() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk_with_size("data", 0xFFFFFFFFu, pcm16_payload(kPcm16Samples)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "unpatched streaming size");
}

// Same unpatched size, but metadata follows the payload. A clamp alone would
// decode the LIST chunk as near-full-scale samples ("LIST"/"INFO" bytes sit at
// ~0.6 amplitude as int16) - a loud burst appended to the reference voice. The
// trailing-chunk scan must cut the audio at the chunk boundary instead.
void test_unpatched_streaming_size_excludes_trailing_chunks() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk_with_size("data", 0xFFFFFFFFu, pcm16_payload(kPcm16Samples)),
        chunk("LIST", std::string("INFOLavf58")),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "streaming size with trailing LIST");
}

// data overstated by a few bytes, swallowing the head of the next chunk. The
// swallowed "LIST" id would otherwise become two garbage samples.
void test_overstated_data_size_excludes_trailing_chunks() {
    const std::string payload = pcm16_payload(kPcm16Samples);
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk_with_size("data", static_cast<uint32_t>(payload.size() + 4), payload),
        chunk("LIST", std::string("INFOLavf58")),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "overstated data with trailing LIST");
}

// The record_windows shape end to end: the declared payload STARTS with the
// tail of an older header (filler + a small fmt chunk + a data header whose
// size runs exactly to EOF), the real PCM follows, and the last bytes of PCM
// sit past the declared range so the walk trips the debris signal. The
// trailing-chunk scan must NOT cut at the embedded old header - its chain
// contains fmt/data, meaning the audio lives inside it, not after it.
void test_embedded_old_header_is_not_a_truncation_point() {
    // 100 zero samples: their 0x00 bytes double as the non-printable debris
    // the walk reads past the declared payload end.
    std::string pcm;
    for (int i = 0; i < 100; ++i) {
        append_le<uint16_t>(pcm, 0u);
    }

    // 36 bytes of old-header debris, mirroring the real files: 2 filler bytes,
    // an 18-byte-body fmt chunk, and a data header pointing exactly at EOF.
    std::string debris(2, '\0');
    std::string old_fmt = fmt_body(1u, 1, 16000, 16);
    append_le<uint16_t>(old_fmt, 0u);  // cbSize: the 18-byte WAVEFORMATEX tail
    debris += chunk("fmt ", old_fmt);
    debris += "data";
    append_le<uint32_t>(debris, static_cast<uint32_t>(pcm.size()));

    // The new header declares debris + PCM minus the 36-byte tail, exactly as
    // record_windows misplaces it.
    const std::string payload = debris + pcm;
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk_with_size(
            "data",
            static_cast<uint32_t>(payload.size() - 36),
            payload.substr(0, payload.size() - 36)),
    }, payload.substr(payload.size() - 36));

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require(
        wav.samples.size() == (payload.size() - 36) / 2,
        "embedded old header must not truncate the audio");
}

// A zero-size data chunk is a placeholder a broken writer left, not the data:
// it must not claim the slot and lock out the real chunk that follows.
void test_zero_size_data_placeholder_does_not_block_the_real_one() {
    const std::string input = build_wav({
        chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
        chunk("data", std::string()),
        chunk("data", pcm16_payload(kPcm16Samples)),
    });

    const auto wav = engine::audio::read_wav_f32(std::string_view(input));
    require_pcm16_samples(wav, "zero-size data placeholder");
}

// --- 1.9 Cases that must keep failing --------------------------------------
void test_incomplete_inputs_still_fail() {
    require_throws(
        build_wav({chunk("data", pcm16_payload(kPcm16Samples))}),
        "WAV without a fmt chunk");

    require_throws(
        build_wav({chunk("fmt ", fmt_body(1u, 1, 16000, 16))}),
        "WAV without a data chunk");

    // A corrupt chunk size before fmt is clamped and skipped, which consumes
    // the rest of the input - the parse then fails as incomplete. (The clamp
    // means no exception fires inside the walk for this shape.)
    require_throws(
        build_wav({
            chunk_with_size("JUNK", 1u << 20, std::string(8, '\0')),
            chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
            chunk("data", pcm16_payload(kPcm16Samples)),
        }),
        "WAV with a corrupt chunk before fmt");

    // A fmt chunk shorter than its 16 mandatory bytes throws inside the walk
    // before anything usable is parsed - the completed-parse tolerance must
    // rethrow this, not swallow it.
    require_throws(
        build_wav({
            chunk("fmt ", fmt_body(1u, 1, 16000, 16).substr(0, 8)),
            chunk("data", pcm16_payload(kPcm16Samples)),
        }),
        "WAV with a malformed fmt chunk");
}

void test_pcm24_from_file() {
    const auto root = std::filesystem::temp_directory_path() / "audio_cpp_wav_reader_test";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(root);
    const auto path = root / "pcm24_stereo.wav";
    write_pcm24_wav(
        path,
        48000,
        2,
        {
            0,
            0x007FFFFF,
            -0x00800000,
            -1,
        });

    const auto wav = engine::audio::read_wav_f32(path);
    require(wav.sample_rate == 48000, "PCM24 sample rate mismatch");
    require(wav.channels == 2, "PCM24 channel count mismatch");
    require(wav.samples.size() == 4, "PCM24 sample count mismatch");
    require_near(wav.samples[0], 0.0F, "PCM24 zero");
    require_near(wav.samples[1], 8388607.0F / 8388608.0F, "PCM24 max positive");
    require_near(wav.samples[2], -1.0F, "PCM24 min negative");
    require_near(wav.samples[3], -1.0F / 8388608.0F, "PCM24 negative one");
}

// Every case runs even when an earlier one fails, so a red run reports the
// whole picture instead of just the first broken layout.
int run_all() {
    const std::vector<std::pair<const char *, void (*)()>> cases = {
        {"pcm24_from_file", test_pcm24_from_file},
        {"extensible_pcm16", test_extensible_pcm16},
        {"extensible_float32", test_extensible_float32},
        {"extensible_truncated_fmt", test_extensible_truncated_fmt},
        {"extensible_truncated_fmt_32bit_is_rejected",
         test_extensible_truncated_fmt_32bit_is_rejected},
        {"later_data_chunk_does_not_replace_the_first",
         test_later_data_chunk_does_not_replace_the_first},
        {"later_fmt_chunk_does_not_replace_the_first",
         test_later_fmt_chunk_does_not_replace_the_first},
        {"extensible_unknown_subformat", test_extensible_unknown_subformat},
        {"junk_and_filler_chunks", test_junk_and_filler_chunks},
        {"trailing_garbage_after_data", test_trailing_garbage_after_data},
        {"truncated_chunk_header_after_data", test_truncated_chunk_header_after_data},
        {"oversized_data_chunk_size", test_oversized_data_chunk_size},
        {"unpatched_streaming_size_reads_to_eof", test_unpatched_streaming_size_reads_to_eof},
        {"unpatched_streaming_size_excludes_trailing_chunks",
         test_unpatched_streaming_size_excludes_trailing_chunks},
        {"overstated_data_size_excludes_trailing_chunks",
         test_overstated_data_size_excludes_trailing_chunks},
        {"zero_size_data_placeholder_does_not_block_the_real_one",
         test_zero_size_data_placeholder_does_not_block_the_real_one},
        {"embedded_old_header_is_not_a_truncation_point",
         test_embedded_old_header_is_not_a_truncation_point},
        {"incomplete_inputs_still_fail", test_incomplete_inputs_still_fail},
    };

    int failures = 0;
    for (const auto & [name, fn] : cases) {
        try {
            fn();
            std::cout << "  [ok]   " << name << "\n";
        } catch (const std::exception & ex) {
            std::cerr << "  [FAIL] " << name << ": " << ex.what() << "\n";
            ++failures;
        }
    }
    return failures;
}

}  // namespace

int main() {
    const int failures = run_all();
    if (failures != 0) {
        std::cerr << "wav_reader_test failed: " << failures << " case(s)\n";
        return 1;
    }
    std::cout << "wav_reader_test passed\n";
    return 0;
}
