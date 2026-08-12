#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace engine::models::irodori_tts {

std::vector<float> build_irodori_duration_features(
    const std::string & text,
    int64_t token_count,
    int64_t max_text_len,
    bool has_speaker);

// Bounds the generated clip to roughly the time the text needs to be spoken.
//
// The duration predictor over-predicts when a reference voice and a caption are
// supplied together, and the model fills the surplus with a phrase that is not
// in the text. Re-running the predictor with the speaker and caption conditions
// disabled removes most of that surplus, but the predictor is itself
// miscalibrated on short text, so a character-count estimate caps it.
struct IrodoriDurationCorrection {
    bool  enabled     = false;
    float text_rate   = 0.207F;  // seconds per spoken codepoint
    float text_margin = 0.4F;    // trailing room left after the text ends
};

// Codepoints in `text` that carry speech time. Punctuation and brackets are
// excluded: they shape the prosody but are not themselves spoken.
int64_t irodori_speech_codepoints(const std::string & text);

// Upper bound on how long `text` takes to speak, from the character rule.
// Returns 0 when the text carries no spoken codepoints.
float irodori_text_duration_estimate(
    const std::string & text,
    const IrodoriDurationCorrection & correction);

// The corrected target length, in seconds.
//
// Takes the smallest of the prediction, the noref prediction and the character
// rule, so the correction can only shorten. Returns `predicted_seconds`
// unchanged when the correction is disabled. Text with no spoken codepoints
// drops the character term only -- the noref prediction still applies.
float irodori_corrected_target_seconds(
    float predicted_seconds,
    float noref_predicted_seconds,
    const std::string & text,
    const IrodoriDurationCorrection & correction);

}  // namespace engine::models::irodori_tts
