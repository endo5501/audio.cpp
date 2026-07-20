#include "engine/framework/audio/audio_reader.h"

#include "engine/framework/audio/wav_writer.h"
#include "engine/framework/io/filesystem.h"

#include "test_assert.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifndef ENGINE_TEST_ASSET_ROOT
#error "ENGINE_TEST_ASSET_ROOT must be defined"
#endif

namespace {

using engine::test::require;

void write_raw(const std::filesystem::path & path, const std::string & bytes) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open test file: " + engine::io::path_to_utf8(path));
    }
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string read_all(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open test file: " + engine::io::path_to_utf8(path));
    }
    return std::string(
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::string message_of_failure(const std::filesystem::path & path) {
    try {
        engine::audio::read_audio_f32(path);
    } catch (const std::exception & ex) {
        return ex.what();
    }
    throw std::runtime_error("expected a failure for: " + engine::io::path_to_utf8(path));
}

void require_contains(
    const std::string & haystack,
    const std::string & needle,
    const std::string & label) {
    if (haystack.find(needle) == std::string::npos) {
        throw std::runtime_error(label + ": expected \"" + needle + "\" in \"" + haystack + "\"");
    }
}

}  // namespace

int main() {
    try {
        const std::filesystem::path asset_root(ENGINE_TEST_ASSET_ROOT);
        const auto mp3_asset = asset_root / "framework" / "audio" / "tone_440hz_16k_mono.mp3";
        require(std::filesystem::exists(mp3_asset),
                "missing MP3 fixture: " + engine::io::path_to_utf8(mp3_asset));

        const auto root = std::filesystem::temp_directory_path() / "audio_cpp_audio_reader_test";
        std::filesystem::remove_all(root);
        std::filesystem::create_directories(root);

        // MP3 reference audio decodes with its own rate and channel layout.
        {
            const auto mp3 = engine::audio::read_audio_f32(mp3_asset);
            require(mp3.sample_rate == 16000, "MP3 sample rate mismatch");
            require(mp3.channels == 1, "MP3 channel count mismatch");
            require(mp3.samples.size() > 1000, "MP3 decoded too few samples");
            bool in_range = true;
            for (const float sample : mp3.samples) {
                if (!(sample >= -1.5F && sample <= 1.5F)) {
                    in_range = false;
                    break;
                }
            }
            require(in_range, "MP3 samples out of the float32 PCM range");
        }

        // WAV still goes through the existing reader unchanged.
        {
            const auto wav_path = root / "tone.wav";
            engine::audio::write_pcm16_wav(wav_path, 24000, 2, {0.0F, 1.0F, -1.0F, 0.25F});
            const auto via_audio = engine::audio::read_audio_f32(wav_path);
            const auto via_wav = engine::audio::read_wav_f32(wav_path);
            require(via_audio.sample_rate == via_wav.sample_rate, "WAV sample rate regression");
            require(via_audio.channels == via_wav.channels, "WAV channel count regression");
            require(via_audio.samples == via_wav.samples, "WAV sample regression");
            require(via_audio.sample_rate == 24000, "WAV sample rate mismatch");
            require(via_audio.channels == 2, "WAV channel count mismatch");
        }

        // Non-ASCII file names must decode; voices/ holds Japanese file names.
        {
            const auto japanese = root / std::filesystem::u8path("月ノ美兎.mp3");
            std::filesystem::copy_file(
                mp3_asset, japanese, std::filesystem::copy_options::overwrite_existing);
            const auto mp3 = engine::audio::read_audio_f32(japanese);
            require(mp3.sample_rate == 16000, "non-ASCII MP3 sample rate mismatch");
            require(mp3.samples.size() > 1000, "non-ASCII MP3 decoded too few samples");
        }

        // A .wav extension declares WAV: MP3 content is rejected, not sniffed.
        {
            const auto mislabeled = root / "actually_mp3.wav";
            std::filesystem::copy_file(
                mp3_asset, mislabeled, std::filesystem::copy_options::overwrite_existing);
            const auto message = message_of_failure(mislabeled);
            require_contains(message, "invalid WAV RIFF header", "mislabeled MP3");
        }

        // Unsupported formats name the file and the supported formats.
        {
            const auto other = root / "notes.txt";
            write_raw(other, "this is not audio at all, just text bytes");
            const auto message = message_of_failure(other);
            require_contains(message, "unsupported audio input format", "unsupported format");
            require_contains(message, "(supported: WAV, MP3)", "unsupported format list");
            require_contains(message, other.string(), "unsupported format path");
        }

        // In-memory inputs (multipart uploads) must sniff the same formats as
        // paths: the server admits .mp3 uploads but never spools them to disk.
        {
            const auto bytes = read_all(mp3_asset);
            const auto mp3 = engine::audio::read_audio_f32(std::string_view(bytes));
            require(mp3.sample_rate == 16000, "in-memory MP3 sample rate mismatch");
            require(mp3.channels == 1, "in-memory MP3 channel count mismatch");
            require(mp3.samples.size() > 1000, "in-memory MP3 decoded too few samples");

            const auto wav_path = root / "buffer.wav";
            engine::audio::write_pcm16_wav(wav_path, 16000, 1, {0.0F, 1.0F, -1.0F, 0.25F});
            const auto wav_bytes = read_all(wav_path);
            const auto wav = engine::audio::read_audio_f32(std::string_view(wav_bytes));
            require(wav.sample_rate == 16000, "in-memory WAV sample rate mismatch");
            require(wav.samples.size() == 4, "in-memory WAV sample count mismatch");

            bool rejected = false;
            try {
                engine::audio::read_audio_f32(std::string_view("not audio at all"));
            } catch (const std::exception & ex) {
                rejected = std::string(ex.what()).find("unsupported audio input format") !=
                    std::string::npos;
            }
            require(rejected, "in-memory garbage must be rejected as unsupported");
        }

        // Error messages cross the C ABI into Dart, which decodes them as UTF-8:
        // a path rendered in the Windows ANSI code page throws FormatException
        // there instead of surfacing the real problem.
        {
            const auto japanese_text = root / std::filesystem::u8path("月ノ美兎.txt");
            write_raw(japanese_text, "this is not audio at all, just text bytes");
            const auto message = message_of_failure(japanese_text);
            const auto utf8 = japanese_text.u8string();
            require_contains(
                message,
                std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size()),
                "non-ASCII path is reported as UTF-8");
        }

        // Empty and corrupt inputs fail with a path, and never crash.
        {
            const auto empty = root / "empty.mp3";
            write_raw(empty, "");
            require_contains(message_of_failure(empty), empty.string(), "empty MP3 path");

            const auto corrupt = root / "corrupt.mp3";
            write_raw(corrupt, std::string(512, '\x01'));
            require_contains(message_of_failure(corrupt), corrupt.string(), "corrupt MP3 path");
        }

        std::cout << "audio_reader_test passed\n";
    } catch (const std::exception & ex) {
        std::cerr << "audio_reader_test failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
