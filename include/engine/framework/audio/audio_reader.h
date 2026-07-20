#pragma once

#include "engine/framework/audio/wav_reader.h"

#include <filesystem>
#include <string_view>

namespace engine::audio {

WavData read_audio_f32(const std::filesystem::path & path);

// Overload for inputs already in memory. Without it a string_view of audio
// bytes would implicitly convert to a path and be read as a file name.
WavData read_audio_f32(std::string_view input);

}  // namespace engine::audio
