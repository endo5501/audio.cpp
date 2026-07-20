#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace engine::io {

// Renders `path` as UTF-8, for messages that leave the process (C ABI errors,
// logs, JSON). path::string() yields the platform's narrow encoding, which on
// Windows is the ANSI code page: a non-ASCII path becomes invalid UTF-8 and
// hosts that decode strictly (the Dart FFI layer among them) report a decode
// failure instead of the error that actually occurred.
std::string path_to_utf8(const std::filesystem::path & path);

bool is_existing_directory(const std::filesystem::path & path);
bool is_existing_file(const std::filesystem::path & path);

std::filesystem::path require_directory(const std::filesystem::path & path, std::string_view role);
std::filesystem::path require_file(const std::filesystem::path & path, std::string_view role);

std::optional<std::filesystem::path> find_first_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::vector<std::filesystem::path> collect_existing(
    const std::filesystem::path & root,
    const std::vector<std::string> & relative_candidates);

std::string read_text_file(const std::filesystem::path & path);

}  // namespace engine::io
