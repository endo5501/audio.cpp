#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <streambuf>

namespace engine::audio {
namespace {

class ReadOnlyMemoryStreamBuffer final : public std::streambuf {
public:
    explicit ReadOnlyMemoryStreamBuffer(std::string_view data) {
        // std::streambuf::setg takes char*, but this input stream only reads from the buffer.
        auto * begin = const_cast<char *>(data.data());
        setg(begin, begin, begin + data.size());
    }

protected:
    pos_type seekoff(off_type offset, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
        if ((which & std::ios_base::in) == 0) {
            return pos_type(off_type(-1));
        }
        char * base = eback();
        char * next = gptr();
        char * end = egptr();
        char * target = nullptr;
        if (dir == std::ios_base::beg) {
            target = base + offset;
        } else if (dir == std::ios_base::cur) {
            target = next + offset;
        } else if (dir == std::ios_base::end) {
            target = end + offset;
        }
        if (target == nullptr || target < base || target > end) {
            return pos_type(off_type(-1));
        }
        setg(base, target, end);
        return pos_type(target - base);
    }

    pos_type seekpos(pos_type position, std::ios_base::openmode which) override {
        return seekoff(off_type(position), std::ios_base::beg, which);
    }
};

template <typename T>
T read_scalar(std::istream & input) {
    T value{};
    input.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("failed to read WAV scalar");
    }
    return value;
}

void skip_bytes(std::istream & input, std::streamoff count) {
    input.seekg(count, std::ios::cur);
    if (!input) {
        throw std::runtime_error("failed to seek inside WAV file");
    }
}

// Bytes left between the current position and the end of the input, used to
// clamp chunk sizes whose header field claims more than the file holds.
std::streamoff remaining_bytes(std::istream & input) {
    const std::streampos current = input.tellg();
    if (current < 0) {
        return 0;
    }
    input.seekg(0, std::ios::end);
    const std::streampos end = input.tellg();
    input.seekg(current, std::ios::beg);
    if (!input || end < current) {
        input.clear();
        input.seekg(current, std::ios::beg);
        return 0;
    }
    return end - current;
}

constexpr uint16_t kWaveFormatPcm = 0x0001;
constexpr uint16_t kWaveFormatIeeeFloat = 0x0003;
constexpr uint16_t kWaveFormatExtensible = 0xFFFE;

// WAVE_FORMAT_EXTENSIBLE keeps the real format tag in the first two bytes of
// the SubFormat GUID. The extension is 22 bytes: cbSize is followed by
// wValidBitsPerSample (2) + dwChannelMask (4) + the 16-byte GUID.
constexpr uint32_t kExtensibleFmtChunkSize = 40;
constexpr std::streamoff kBytesBeforeSubFormatGuid = 8;

// A truncated extension is not fatal: some writers copy only the 18-byte
// WAVEFORMATEX and leave the GUID out, so fall back to the bit depth.
uint16_t format_from_bits_per_sample(uint16_t bits_per_sample) {
    return bits_per_sample == 32 ? kWaveFormatIeeeFloat : kWaveFormatPcm;
}

void parse_fmt_chunk(
    std::istream & input,
    std::streamoff chunk_size,
    uint16_t & audio_format,
    uint16_t & channels,
    uint32_t & sample_rate,
    uint16_t & bits_per_sample) {
    if (chunk_size < 16) {
        throw std::runtime_error("malformed WAV fmt chunk");
    }

    audio_format = read_scalar<uint16_t>(input);
    channels = read_scalar<uint16_t>(input);
    sample_rate = read_scalar<uint32_t>(input);
    skip_bytes(input, 6);
    bits_per_sample = read_scalar<uint16_t>(input);
    std::streamoff consumed = 16;

    if (audio_format == kWaveFormatExtensible) {
        if (chunk_size >= static_cast<std::streamoff>(kExtensibleFmtChunkSize)) {
            skip_bytes(input, kBytesBeforeSubFormatGuid);
            audio_format = read_scalar<uint16_t>(input);
            consumed += kBytesBeforeSubFormatGuid + 2;
        } else {
            audio_format = format_from_bits_per_sample(bits_per_sample);
        }
    }

    if (chunk_size > consumed) {
        skip_bytes(input, chunk_size - consumed);
    }
}

}  // namespace

WavData read_wav_f32(std::istream & input) {
    if (!input) {
        throw std::runtime_error("could not open WAV input");
    }

    char riff[4];
    input.read(riff, 4);
    if (!input || std::string(riff, 4) != "RIFF") {
        throw std::runtime_error("invalid WAV RIFF header");
    }
    skip_bytes(input, 4);
    char wave[4];
    input.read(wave, 4);
    if (!input || std::string(wave, 4) != "WAVE") {
        throw std::runtime_error("invalid WAV WAVE header");
    }

    uint16_t audio_format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::vector<char> data;

    bool have_fmt = false;
    bool have_data = false;

    while (input) {
        char chunk_id[4];
        input.read(chunk_id, 4);
        if (input.gcount() != 4) {
            break;
        }
        const std::string id(chunk_id, 4);

        // Once fmt and data are both in hand the audio is recoverable, so
        // damage found further along stops the walk instead of discarding what
        // was already parsed. Writers do leave debris behind the data chunk:
        // record_windows overwrites its sink writer's header with a shorter one
        // and leaves the tail of the old header in the file.
        try {
            const uint32_t chunk_size = read_scalar<uint32_t>(input);
            // A size field that claims more than the file holds is a header
            // defect, not a reason to give up on the samples that are there.
            const std::streamoff size = std::min<std::streamoff>(
                static_cast<std::streamoff>(chunk_size), remaining_bytes(input));

            if (id == "fmt ") {
                parse_fmt_chunk(input, size, audio_format, channels, sample_rate, bits_per_sample);
                have_fmt = true;
            } else if (id == "data") {
                data.resize(static_cast<size_t>(size));
                if (size > 0) {
                    input.read(data.data(), static_cast<std::streamsize>(size));
                    if (input.gcount() != size) {
                        throw std::runtime_error("failed to read WAV data chunk");
                    }
                }
                have_data = true;
            } else {
                skip_bytes(input, size);
            }
            if (chunk_size % 2 == 1) {
                skip_bytes(input, 1);
            }
        } catch (const std::runtime_error &) {
            if (!have_fmt || !have_data) {
                throw;
            }
            break;
        }
    }

    if (channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data.empty()) {
        throw std::runtime_error("incomplete WAV file");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(channels);

    if (audio_format == 1 && bits_per_sample == 16) {
        const size_t sample_count = data.size() / sizeof(int16_t);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const int16_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
        }
        return wav;
    }

    if (audio_format == 1 && bits_per_sample == 24) {
        if (data.size() % 3 != 0) {
            throw std::runtime_error("malformed PCM24 WAV data chunk");
        }
        const size_t sample_count = data.size() / 3;
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const uint8_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            const size_t offset = i * 3;
            int32_t value =
                static_cast<int32_t>(pcm[offset]) |
                (static_cast<int32_t>(pcm[offset + 1]) << 8) |
                (static_cast<int32_t>(pcm[offset + 2]) << 16);
            if ((value & 0x00800000) != 0) {
                value |= ~0x00FFFFFF;
            }
            wav.samples[i] = static_cast<float>(value) / 8388608.0F;
        }
        return wav;
    }

    if (audio_format == 3 && bits_per_sample == 32) {
        const size_t sample_count = data.size() / sizeof(float);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const float *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = pcm[i];
        }
        return wav;
    }

    throw std::runtime_error("unsupported WAV encoding (need PCM16, PCM24, or float32)");
}

WavData read_wav_f32(std::string_view input) {
    ReadOnlyMemoryStreamBuffer buffer(input);
    std::istream stream(&buffer);
    return read_wav_f32(stream);
}

WavData read_wav_f32(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("could not open WAV input: " + path.string());
    }

    return read_wav_f32(input);
}

}  // namespace engine::audio
