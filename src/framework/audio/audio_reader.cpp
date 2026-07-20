#include "engine/framework/audio/audio_reader.h"

#include "engine/framework/io/filesystem.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#define MINIMP3_FLOAT_OUTPUT
#define MINIMP3_IMPLEMENTATION
#include "minimp3_ex.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace engine::audio {
namespace {

constexpr const char * kSupportedFormats = " (supported: WAV, MP3)";

std::string lower_extension(const std::filesystem::path & path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return ext;
}

std::string read_file_bytes(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open audio input: " + io::path_to_utf8(path));
    }
    input.seekg(0, std::ios::end);
    const auto size = input.tellg();
    if (size < 0) {
        throw std::runtime_error("failed to size audio input: " + io::path_to_utf8(path));
    }
    input.seekg(0, std::ios::beg);
    std::string data(static_cast<size_t>(size), '\0');
    if (!data.empty()) {
        input.read(data.data(), static_cast<std::streamsize>(data.size()));
        if (!input) {
            throw std::runtime_error("failed to read audio input: " + io::path_to_utf8(path));
        }
    }
    return data;
}

bool has_wav_header(std::string_view data) {
    return data.size() >= 12 && std::memcmp(data.data(), "RIFF", 4) == 0 &&
        std::memcmp(data.data() + 8, "WAVE", 4) == 0;
}

bool is_mp3_data(std::string_view data) {
    return mp3dec_detect_buf(reinterpret_cast<const uint8_t *>(data.data()), data.size()) == 0;
}

// `label` names the input in error messages: a UTF-8 path for file inputs, or a
// generic description for in-memory buffers, which have no name to report.
WavData read_mp3_f32(const std::string & label, std::string_view data) {
    if (data.empty()) {
        throw std::runtime_error("empty MP3 input: " + label);
    }

    mp3dec_t decoder;
    mp3dec_init(&decoder);
    mp3dec_file_info_t info{};
    const int rc = mp3dec_load_buf(
        &decoder, reinterpret_cast<const uint8_t *>(data.data()), data.size(), &info, nullptr,
        nullptr);
    if (rc != 0) {
        throw std::runtime_error("failed to decode MP3 input: " + label);
    }
    if (info.buffer == nullptr || info.samples == 0 || info.hz <= 0 || info.channels <= 0) {
        std::free(info.buffer);
        throw std::runtime_error("decoded MP3 input is empty: " + label);
    }

    WavData audio;
    audio.sample_rate = info.hz;
    audio.channels = info.channels;
    audio.samples.assign(info.buffer, info.buffer + info.samples);
    std::free(info.buffer);
    return audio;
}

}  // namespace

bool is_supported_audio_extension(std::string_view extension) {
    return extension == ".wav" || extension == ".mp3" || extension == ".mpa" ||
        extension == ".mpeg";
}

WavData read_audio_f32(const std::filesystem::path & path) {
    const std::string label = io::path_to_utf8(path);
    const auto ext = lower_extension(path);
    // An extension the decoder does not know is rejected before the file is
    // read: no point loading megabytes to discover the format is unsupported.
    if (!ext.empty() && !is_supported_audio_extension(ext)) {
        throw std::runtime_error("unsupported audio input format: " + label + kSupportedFormats);
    }

    const std::string data = read_file_bytes(path);
    if (data.empty()) {
        throw std::runtime_error("empty audio input: " + label);
    }
    if (has_wav_header(data)) {
        // Explicit string_view: a std::string is convertible to path as well,
        // so the overload set is ambiguous without it.
        return read_wav_f32(std::string_view(data));
    }
    // A .wav extension is a declaration, not a hint: content that is not RIFF
    // is a broken WAV rather than something to sniff further.
    if (ext == ".wav") {
        throw std::runtime_error("invalid WAV RIFF header: " + label);
    }
    if (!ext.empty() || is_mp3_data(data)) {
        return read_mp3_f32(label, data);
    }
    throw std::runtime_error("unsupported audio input format: " + label + kSupportedFormats);
}

// In-memory inputs (multipart uploads) never reach the filesystem, so there is
// no extension to consult: the format is decided by the leading bytes alone.
WavData read_audio_f32(std::string_view input) {
    if (input.empty()) {
        throw std::runtime_error("empty audio input");
    }
    if (has_wav_header(input)) {
        return read_wav_f32(input);
    }
    if (is_mp3_data(input)) {
        return read_mp3_f32("in-memory audio input", input);
    }
    throw std::runtime_error(std::string("unsupported audio input format") + kSupportedFormats);
}

}  // namespace engine::audio
