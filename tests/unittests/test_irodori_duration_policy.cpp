#include "engine/models/irodori_tts/duration.h"

#include "test_assert.h"

#include <iostream>
#include <string>

// The numbers below are the measured values that motivated the correction.
// Reference + caption makes the duration predictor ask for more time than the
// RF body needs, and the model fills the surplus with an unrequested phrase.
// Measured on Irodori-TTS-v4-Small, three texts, listening-verified:
//
//   text        chars(-punct)  predicted  noref predicted  actual speech end
//   32 chars    29             7.96       5.56             5.49
//   12 chars    10             4.60       3.76             2.05
//   54 chars    50             11.64      11.20            10.35
//
// The noref prediction is accurate for the medium text but over-predicts a
// short one by +83%, so neither the noref prediction nor the character rule
// alone covers every case -- the correction takes the minimum of both.
int main() try {
  using engine::models::irodori_tts::IrodoriDurationCorrection;
  using engine::models::irodori_tts::irodori_corrected_target_seconds;
  using engine::models::irodori_tts::irodori_speech_codepoints;
  using engine::models::irodori_tts::irodori_text_duration_estimate;

  const std::string medium =
      u8"どうしてもっと早く教えてくれなかったの？私、ずっと待ってたのに。";
  const std::string short_text = u8"こんにちは、テストです。";
  const std::string long_text =
      u8"窓の外では、粉雪が音もなく降り積もっていた。彼女はカーテンの隙間から"
      u8"その白い景色を、ただ黙って見つめている。";

  IrodoriDurationCorrection enabled;
  enabled.enabled = true;

  IrodoriDurationCorrection disabled;

  // Punctuation does not take time to speak, so it is excluded from the count.
  engine::test::require_eq(irodori_speech_codepoints(medium), int64_t{29},
                           "medium codepoints");
  engine::test::require_eq(irodori_speech_codepoints(short_text), int64_t{10},
                           "short codepoints");
  engine::test::require_eq(irodori_speech_codepoints(long_text), int64_t{50},
                           "long codepoints");
  engine::test::require_eq(irodori_speech_codepoints(""), int64_t{0},
                           "empty codepoints");

  // 29 * 0.207 + 0.4
  engine::test::require_close(irodori_text_duration_estimate(medium, enabled),
                              6.403F, 0.001F, "medium text estimate");
  engine::test::require_close(
      irodori_text_duration_estimate(short_text, enabled), 2.470F, 0.001F,
      "short text estimate");
  engine::test::require_close(irodori_text_duration_estimate(long_text, enabled),
                              10.750F, 0.001F, "long text estimate");

  // Medium text: the noref prediction is the smallest of the three.
  engine::test::require_close(
      irodori_corrected_target_seconds(7.96F, 5.56F, medium, enabled), 5.56F,
      0.001F, "medium picks noref prediction");

  // Short text: the noref prediction over-predicts, the character rule wins.
  engine::test::require_close(
      irodori_corrected_target_seconds(4.60F, 3.76F, short_text, enabled),
      2.470F, 0.001F, "short picks character rule");

  // Long text: the character rule wins again.
  engine::test::require_close(
      irodori_corrected_target_seconds(11.64F, 11.20F, long_text, enabled),
      10.750F, 0.001F, "long picks character rule");

  // The correction never lengthens: a prediction below both other terms stands.
  engine::test::require_close(
      irodori_corrected_target_seconds(4.00F, 5.56F, medium, enabled), 4.00F,
      0.001F, "correction never lengthens");

  // Disabled is a no-op, whatever the other terms say.
  engine::test::require_close(
      irodori_corrected_target_seconds(7.96F, 5.56F, medium, disabled), 7.96F,
      0.001F, "disabled keeps the prediction");

  // A text of only punctuation has no speech to bound; the character rule must
  // not collapse the target to the margin and truncate whatever is generated.
  engine::test::require_close(
      irodori_corrected_target_seconds(3.00F, 2.50F, u8"、。", enabled), 2.50F,
      0.001F, "punctuation-only text falls back to the predictions");

  // A negative rate or margin would make the character rule the smallest term
  // and collapse the target, truncating the clip. Reject rather than generate a
  // silently wrong length.
  {
    IrodoriDurationCorrection bad_rate = enabled;
    bad_rate.text_rate = -0.1F;
    bool threw = false;
    try {
      irodori_corrected_target_seconds(7.96F, 5.56F, medium, bad_rate);
    } catch (const std::exception &) {
      threw = true;
    }
    engine::test::require(threw, "negative rate must be rejected");
  }
  {
    IrodoriDurationCorrection bad_margin = enabled;
    bad_margin.text_margin = -1.0F;
    bool threw = false;
    try {
      irodori_corrected_target_seconds(7.96F, 5.56F, medium, bad_margin);
    } catch (const std::exception &) {
      threw = true;
    }
    engine::test::require(threw, "negative margin must be rejected");
  }

  // A disabled correction is inert, so it does not validate its parameters.
  {
    IrodoriDurationCorrection bad_but_off = disabled;
    bad_but_off.text_rate = -0.1F;
    engine::test::require_close(
        irodori_corrected_target_seconds(7.96F, 5.56F, medium, bad_but_off),
        7.96F, 0.001F, "disabled ignores its parameters");
  }

  std::cout << "irodori_duration_policy_test passed\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "irodori_duration_policy_test failed: " << error.what() << "\n";
  return 1;
}
