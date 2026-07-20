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
    // walks into it and reads a bogus id plus an enormous size.
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

// --- 1.9 Cases that must keep failing --------------------------------------
void test_incomplete_inputs_still_fail() {
    require_throws(
        build_wav({chunk("data", pcm16_payload(kPcm16Samples))}),
        "WAV without a fmt chunk");

    require_throws(
        build_wav({chunk("fmt ", fmt_body(1u, 1, 16000, 16))}),
        "WAV without a data chunk");

    // A corrupt chunk size before fmt: nothing usable has been parsed yet, so
    // this must not be swallowed by the trailing-corruption tolerance.
    require_throws(
        build_wav({
            chunk_with_size("JUNK", 1u << 20, std::string(8, '\0')),
            chunk("fmt ", fmt_body(1u, 1, 16000, 16)),
            chunk("data", pcm16_payload(kPcm16Samples)),
        }),
        "WAV with a corrupt chunk before fmt");
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
        {"extensible_unknown_subformat", test_extensible_unknown_subformat},
        {"junk_and_filler_chunks", test_junk_and_filler_chunks},
        {"trailing_garbage_after_data", test_trailing_garbage_after_data},
        {"oversized_data_chunk_size", test_oversized_data_chunk_size},
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
