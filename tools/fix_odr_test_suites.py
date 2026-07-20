from pathlib import Path

path = Path(r"d:\_phoenix\_079\v6.0Alixander\phoenix\tests\gtest\unit\emotion\test_llamacpp_emotion_adjuster.cpp")
text = path.read_text(encoding="utf-8")

text = text.replace("TEST(EmotionWeightCacheTest,", "TEST(EmotionWeightCacheFunctions,")
text = text.replace("TEST(LlamaCppEmotionWeightAdjusterTest,", "TEST(LlamaCppEmotionWeightAdjusterFunctions,")

path.write_text(text, encoding="utf-8")
print("renamed free test suites in test_llamacpp_emotion_adjuster.cpp")
