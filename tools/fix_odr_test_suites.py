from pathlib import Path

path = Path(__file__).resolve().parents[1] / "tests" / "gtest" / "unit" / "emotion" / "test_llamacpp_emotion_adjuster.cpp"
text = path.read_text(encoding="utf-8")

text = text.replace("TEST(EmotionWeightCacheTest,", "TEST(EmotionWeightCacheFunctions,")
text = text.replace("TEST(LlamaCppEmotionWeightAdjusterTest,", "TEST(LlamaCppEmotionWeightAdjusterFunctions,")

path.write_text(text, encoding="utf-8")
print("renamed free test suites in test_llamacpp_emotion_adjuster.cpp")
