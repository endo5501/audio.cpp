#include "engine/framework/audio/wav_reader.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
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
//
// Returns 0 when the position cannot be determined. That conflation is safe
// only because this reader already requires a seekable stream — the RIFF size
// field is stepped over with a seek before any chunk is read, so a pipe-backed
// stream fails there and never reaches this point.
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
//
// 32 bits is the exception. Integer PCM32 and IEEE float32 share the layout,
// and only the GUID tells them apart, so guessing would reinterpret integer
// samples as floats and emit garbage. Leave the tag as EXTENSIBLE there so it
// falls through to the unsupported-encoding error instead.
uint16_t format_from_bits_per_sample(uint16_t bits_per_sample) {
    if (bits_per_sample == 16 || bits_per_sample == 24) {
        return kWaveFormatPcm;
    }
    return kWaveFormatExtensible;
}

// RIFF ids are four printable-ASCII bytes. Anything else after the payload is
// debris, not a chunk - the signal that some size field upstream was wrong.
bool is_plausible_chunk_id(const char * id) {
    for (int i = 0; i < 4; ++i) {
        const unsigned char c = static_cast<unsigned char>(id[i]);
        if (c < 0x20 || c > 0x7E) {
            return false;
        }
    }
    return true;
}

// Reads a chunk header at an absolute input offset, out of the data buffer
// when the bytes are there (they usually are - the bad size read them in as
// "audio") and from the stream when the header straddles the buffer's end.
bool read_chunk_header_at(
    std::istream & input,
    const std::vector<char> & data,
    std::streamoff data_offset,
    std::streamoff pos,
    char (&id)[4],
    uint32_t & size) {
    const std::streamoff rel = pos - data_offset;
    if (rel >= 0 && rel + 8 <= static_cast<std::streamoff>(data.size())) {
        std::memcpy(id, data.data() + rel, 4);
        std::memcpy(&size, data.data() + rel + 4, 4);
        return true;
    }
    input.clear();
    input.seekg(pos, std::ios::beg);
    input.read(id, 4);
    if (input.gcount() != 4) {
        return false;
    }
    input.read(reinterpret_cast<char *>(&size), 4);
    return input.gcount() == 4;
}

// True when a chain of plausible chunks starting at `pos` walks exactly to the
// end of the input. Exactness is the safety net: a random 4-printable-byte run
// inside real audio almost never carries a size that lands on EOF, so false
// positives - which would truncate genuine samples - stay negligible, and the
// scan only runs on files whose data size already proved unreliable.
bool chunk_chain_reaches_end(
    std::istream & input,
    const std::vector<char> & data,
    std::streamoff data_offset,
    std::streamoff pos,
    std::streamoff input_end) {
    while (pos + 8 <= input_end) {
        char id[4];
        uint32_t size = 0;
        if (!read_chunk_header_at(input, data, data_offset, pos, id, size)) {
            return false;
        }
        if (!is_plausible_chunk_id(id)) {
            return false;
        }
        // A chain carrying fmt or data is not trailing metadata - it is an
        // older header embedded at the head of the payload (record_windows
        // buries its sink writer's header there, and its data chunk points
        // exactly at EOF). Truncating at such a chain would cut away the
        // audio itself, so it is never a truncation point.
        if (std::memcmp(id, "fmt ", 4) == 0 || std::memcmp(id, "data", 4) == 0) {
            return false;
        }
        const std::streamoff next = pos + 8 + static_cast<std::streamoff>(size);
        if (next == input_end) {
            return true;
        }
        // RIFF pads odd-sized chunks; accept the padded end too.
        const std::streamoff padded = next + (size % 2);
        if (padded == input_end) {
            return true;
        }
        if (padded > input_end) {
            return false;
        }
        pos = padded;
    }
    return false;
}

// When the data chunk's declared size proved unreliable (it was clamped, or
// the walk hit debris right after it), bytes at the tail of `data` may really
// be trailing chunks (LIST/INFO, id3, ...) that the bad size swallowed. Those
// decode as near-full-scale garbage samples, so find the first position whose
// chunk chain runs exactly to EOF and cut the audio there.
void truncate_swallowed_trailing_chunks(
    std::istream & input,
    std::vector<char> & data,
    std::streamoff data_offset) {
    input.clear();
    input.seekg(0, std::ios::end);
    const std::streamoff input_end = input.tellg();
    if (input_end < 0) {
        return;
    }
    // A swallowed chain necessarily starts inside what was read as audio.
    for (size_t p = 0; p < data.size(); ++p) {
        if (chunk_chain_reaches_end(
                input, data, data_offset,
                data_offset + static_cast<std::streamoff>(p), input_end)) {
            data.resize(p);
            return;
        }
    }
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
    // Set when the data chunk's size field proved unreliable: it was clamped,
    // or the walk ran into debris right behind the payload. Either way the
    // tail of `data` may hold swallowed trailing chunks rather than audio.
    bool data_size_untrusted = false;
    std::streamoff data_payload_offset = -1;

    while (input) {
        char chunk_id[4];
        input.read(chunk_id, 4);
        if (input.gcount() != 4) {
            break;
        }
        const std::string id(chunk_id, 4);

        // With fmt and data both in hand, a non-printable id is not a chunk -
        // it is debris (record_windows leaves the tail of its sink writer's
        // old header behind the payload). Stop the walk and flag the size as
        // suspect instead of decoding garbage sizes.
        if (have_fmt && have_data && !is_plausible_chunk_id(chunk_id)) {
            data_size_untrusted = true;
            break;
        }

        // Once fmt and data are both in hand the audio is recoverable, so
        // damage found further along stops the walk instead of discarding what
        // was already parsed.
        try {
            const uint32_t chunk_size = read_scalar<uint32_t>(input);
            // A size field that claims more than the file holds is a header
            // defect, not a reason to give up on the samples that are there.
            const std::streamoff size = std::min<std::streamoff>(
                static_cast<std::streamoff>(chunk_size), remaining_bytes(input));

            // A second fmt or data chunk is not a layout any real WAV uses, so
            // it is debris that happens to spell the right four bytes. Keeping
            // the first one stops it from truncating audio already read.
            if (id == "fmt " && !have_fmt) {
                parse_fmt_chunk(input, size, audio_format, channels, sample_rate, bits_per_sample);
                have_fmt = true;
            } else if (id == "data" && !have_data) {
                if (static_cast<std::streamoff>(chunk_size) != size) {
                    data_size_untrusted = true;
                }
                data_payload_offset = input.tellg();
                data.resize(static_cast<size_t>(size));
                if (size > 0) {
                    input.read(data.data(), static_cast<std::streamsize>(size));
                    if (input.gcount() != size) {
                        throw std::runtime_error("failed to read WAV data chunk");
                    }
                }
                // A zero-size chunk is a placeholder a broken writer left, not
                // the data: leave the slot open for the real one.
                have_data = !data.empty();
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
            data_size_untrusted = true;
            break;
        }
    }

    if (data_size_untrusted && !data.empty() && data_payload_offset >= 0) {
        truncate_swallowed_trailing_chunks(input, data, data_payload_offset);
    }

    if (channels == 0 || sample_rate == 0 || bits_per_sample == 0 || data.empty()) {
        throw std::runtime_error("incomplete WAV file");
    }

    WavData wav;
    wav.sample_rate = static_cast<int>(sample_rate);
    wav.channels = static_cast<int>(channels);

    if (audio_format == kWaveFormatPcm && bits_per_sample == 16) {
        const size_t sample_count = data.size() / sizeof(int16_t);
        wav.samples.resize(sample_count);
        const auto * pcm = reinterpret_cast<const int16_t *>(data.data());
        for (size_t i = 0; i < sample_count; ++i) {
            wav.samples[i] = static_cast<float>(pcm[i]) / 32768.0F;
        }
        return wav;
    }

    if (audio_format == kWaveFormatPcm && bits_per_sample == 24) {
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

    if (audio_format == kWaveFormatIeeeFloat && bits_per_sample == 32) {
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
