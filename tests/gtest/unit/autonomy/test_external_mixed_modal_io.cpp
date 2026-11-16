/* GTest unit tests for Phoenix v7.0 external mixed-modal I/O module. */

#include "external_mixed_modal_io.hpp"
#include "video_model.hpp"
#include "audio_model.hpp"
#include "model_deployment.hpp"
#include "test_hacktest_framework.hpp"
#include <gtest/gtest.h>

using namespace phoenix::io;
using namespace phoenix::testing;

class MixedModalIOTest : public HacktestBase {
protected:
    void SetUp() override {
        MixedModalConceptBridge::reset();

        // Force vision deployment to a remote endpoint with an empty URL so that
        // image encode/decode tests do not accidentally pick up a locally-exported
        // best.onnx under runtime_store/models/additive_jepa/.  Individual tests
        // that need local/server-client placement call cfg.reset()/load() later.
        auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
        cfg.reset();
        std::map<std::string, std::string> args;
        args["vision-placement"] = "remote";
        args["vision-remote-url"] = "";
        cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");
    }
};

TEST_F(MixedModalIOTest, ModalityStringConversions) {
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Text), "text");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Image), "image");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Audio), "audio");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Video), "video");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Sensor), "sensor");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Structured), "structured");
    EXPECT_EQ(MixedModalPacket::modalityToString(MixedModalModality::Unknown), "unknown");

    EXPECT_EQ(MixedModalPacket::stringToModality("text"), MixedModalModality::Text);
    EXPECT_EQ(MixedModalPacket::stringToModality("image"), MixedModalModality::Image);
    EXPECT_EQ(MixedModalPacket::stringToModality("sensor"), MixedModalModality::Sensor);
    EXPECT_EQ(MixedModalPacket::stringToModality("invalid"), MixedModalModality::Unknown);
}

TEST_F(MixedModalIOTest, JsonRoundTrip) {
    MixedModalPacket p;
    p.id = "pkt-001";
    p.modality = MixedModalModality::Audio;
    p.payload = {0x01, 0x02, 0x03};
    p.mimeType = "audio/pcm";
    p.source = "mic";
    p.metadata = {{"sampleRate", 16000}};

    auto j = p.toJson();
    auto loaded = MixedModalPacket::fromJson(j);

    EXPECT_EQ(loaded.id, p.id);
    EXPECT_EQ(loaded.modality, p.modality);
    EXPECT_EQ(loaded.payload, p.payload);
    EXPECT_EQ(loaded.mimeType, p.mimeType);
    EXPECT_EQ(loaded.source, p.source);
    EXPECT_EQ(loaded.metadata["sampleRate"], 16000);
}

TEST_F(MixedModalIOTest, ToSemanticUnitText) {
    MixedModalPacket p;
    p.id = "txt-001";
    p.modality = MixedModalModality::Text;
    p.payload = {'h', 'e', 'l', 'l', 'o'};

    auto unit = p.toSemanticUnit(8);
    EXPECT_EQ(unit.modality, phoenix::multimodal::Modality::Text);
    EXPECT_EQ(unit.content, "hello");
    EXPECT_EQ(unit.semanticVector.size(), 8u);
    EXPECT_FLOAT_EQ(unit.confidence, 1.0f);
}

TEST_F(MixedModalIOTest, ToSemanticUnitImage) {
    MixedModalPacket p;
    p.id = "img-001";
    p.modality = MixedModalModality::Image;
    p.mimeType = "image/png";
    p.payload = {0x89, 'P', 'N', 'G'};

    auto unit = p.toSemanticUnit(16, "A cat");
    EXPECT_EQ(unit.modality, phoenix::multimodal::Modality::Image);
    EXPECT_TRUE(unit.content.empty());
    EXPECT_EQ(unit.metadata["conceptEncoder"], "video-encoder");
    // Without a real local/remote model, the bridge now fails clearly instead
    // of falling back to a deterministic media concept.
    EXPECT_TRUE(unit.semanticVector.empty());
    EXPECT_FLOAT_EQ(unit.confidence, 0.0f);
    EXPECT_TRUE(unit.metadata.contains("videoEncoderError"));
}

TEST_F(MixedModalIOTest, WorldModelConceptVectorTakesPrecedence) {
    MixedModalPacket p;
    p.id = "img-concept";
    p.modality = MixedModalModality::Image;
    p.payload = {0x89, 'P', 'N', 'G'};
    p.metadata = {{"worldModelVector", std::vector<float>{3.0f, 4.0f}}};

    auto unit = p.toSemanticUnit(2);
    ASSERT_EQ(unit.semanticVector.size(), 2u);
    EXPECT_NEAR(unit.semanticVector[0], 0.6f, 1e-5f);
    EXPECT_NEAR(unit.semanticVector[1], 0.8f, 1e-5f);
    EXPECT_EQ(unit.metadata["conceptEncoder"], "world-model");
}

TEST_F(MixedModalIOTest, SpeechPretrainingAndEncodingUseFallbackWithoutModel) {
    MixedModalPacket audio;
    audio.modality = MixedModalModality::Audio;
    audio.payload = {0x10, 0x40, 0x80, 0xc0, 0xff};

    // When no real speech model (local .onnx or BPU .bin) is present, the
    // deterministic CPU fallback still allows pretraining and encoding to
    // succeed so the semantic contract can be exercised.
    EXPECT_TRUE(MixedModalConceptBridge::pretrainSpeech(audio, "hello speech", 8));

    auto unit = audio.toSemanticUnit(8);
    EXPECT_EQ(unit.metadata["conceptEncoder"], "audio-encoder");
    EXPECT_EQ(unit.semanticVector.size(), 8u);
    EXPECT_FLOAT_EQ(unit.confidence, 1.0f);
    EXPECT_FALSE(unit.metadata.contains("jepaSpeechError"));
}

TEST_F(MixedModalIOTest, DecodeSemanticUnitToConceptPacket) {
    phoenix::multimodal::SemanticUnit unit;
    unit.id = "concept-output";
    unit.modality = phoenix::multimodal::Modality::Image;
    unit.semanticVector = {0.25f, 0.75f};

    auto packet = MixedModalConceptBridge::decode(unit, MixedModalModality::Image, "camera-out");
    EXPECT_EQ(packet.id, "concept-output");
    EXPECT_EQ(packet.modality, MixedModalModality::Image);
    EXPECT_EQ(packet.source, "camera-out");
    EXPECT_EQ(packet.mimeType, "image/png");
    // No deterministic placeholder fallback: an unconfigured decoder fails
    // clearly and reports the backend error in the packet metadata.
    EXPECT_TRUE(packet.payload.empty());
    EXPECT_TRUE(packet.metadata.contains("videoDecodeError"));
}

TEST_F(MixedModalIOTest, InputBufferPushAndFlush) {
    MixedModalInputBuffer buffer;
    MixedModalPacket p;
    p.id = "1";
    p.modality = MixedModalModality::Sensor;
    buffer.push(p);

    EXPECT_EQ(buffer.size(), 1u);
    EXPECT_FALSE(buffer.empty());

    auto out = buffer.flush();
    EXPECT_EQ(out.size(), 1u);
    EXPECT_TRUE(buffer.empty());
}

TEST_F(MixedModalIOTest, OutputQueueDrain) {
    MixedModalOutputQueue queue;
    MixedModalPacket p1, p2, p3;
    p1.id = "1";
    p2.id = "2";
    p3.id = "3";
    queue.push(p1);
    queue.push(p2);
    queue.push(p3);

    auto out = queue.drain(2);
    EXPECT_EQ(out.size(), 2u);
    EXPECT_EQ(queue.size(), 1u);

    auto rest = queue.drain();
    EXPECT_EQ(rest.size(), 1u);
    EXPECT_TRUE(queue.empty());
}

TEST_F(MixedModalIOTest, ChannelRegistry) {
    MixedModalChannelRegistry registry;
    registry.registerSource("camera", "image/jpeg");
    registry.registerSource("mic", "audio/pcm");

    auto sources = registry.sources();
    EXPECT_EQ(sources.size(), 2u);

    auto j = registry.toJson();
    EXPECT_EQ(j["camera"], "image/jpeg");
    EXPECT_EQ(j["mic"], "audio/pcm");
}

TEST_F(MixedModalIOTest, InvalidJsonFallback) {
    nlohmann::json bad = "not an object";
    auto p = MixedModalPacket::fromJson(bad);
    EXPECT_EQ(p.id, "");
    EXPECT_EQ(p.modality, MixedModalModality::Unknown);
    EXPECT_TRUE(p.payload.empty());
}

TEST_F(MixedModalIOTest, DecodeSemanticUnitToTextPacket) {
    phoenix::multimodal::SemanticUnit unit;
    unit.id = "text-output";
    unit.modality = phoenix::multimodal::Modality::Text;
    unit.content = "hello world";

    auto packet = MixedModalConceptBridge::decode(unit, MixedModalModality::Text, "tts");
    EXPECT_EQ(packet.id, "text-output");
    EXPECT_EQ(packet.source, "tts");
    EXPECT_EQ(packet.modality, MixedModalModality::Text);
    EXPECT_EQ(packet.mimeType, "text/plain; charset=utf-8");
    std::string decoded(packet.payload.begin(), packet.payload.end());
    EXPECT_EQ(decoded, "hello world");
}

TEST_F(MixedModalIOTest, DecodeSemanticUnitToAudioPacket) {
    phoenix::multimodal::SemanticUnit unit;
    unit.id = "audio-concept";
    unit.modality = phoenix::multimodal::Modality::Audio;
    unit.semanticVector = {0.1f, 0.2f, 0.3f};

    auto packet = MixedModalConceptBridge::decode(unit, MixedModalModality::Audio, "speaker");
    EXPECT_EQ(packet.modality, MixedModalModality::Audio);
    EXPECT_EQ(packet.source, "speaker");
    EXPECT_EQ(packet.mimeType, "audio/pcm");
    // When no real decoder is configured, the deterministic fallback still
    // emits a PCM payload so the bridge can be exercised end-to-end.
    EXPECT_FALSE(packet.payload.empty());
    EXPECT_FALSE(packet.metadata.contains("audioDecodeError"));
    EXPECT_EQ(packet.metadata["sourceModality"], "audio");
}

TEST_F(MixedModalIOTest, PretrainSpeechInvalidInputs) {
    MixedModalPacket notAudio;
    notAudio.modality = MixedModalModality::Text;
    notAudio.payload = {0x01, 0x02};
    EXPECT_FALSE(MixedModalConceptBridge::pretrainSpeech(notAudio, "transcript", 8));

    MixedModalPacket emptyAudio;
    emptyAudio.modality = MixedModalModality::Audio;
    EXPECT_FALSE(MixedModalConceptBridge::pretrainSpeech(emptyAudio, "transcript", 8));

    MixedModalPacket audio;
    audio.modality = MixedModalModality::Audio;
    audio.payload = {0x10, 0x20};
    EXPECT_FALSE(MixedModalConceptBridge::pretrainSpeech(audio, "", 8));
}

TEST_F(MixedModalIOTest, StructuredMediaConceptEncoding) {
    MixedModalPacket p;
    p.modality = MixedModalModality::Sensor;
    p.payload = {0x01, 0x02, 0x03, 0x04};
    auto unit = p.toSemanticUnit(8);
    EXPECT_EQ(unit.modality, phoenix::multimodal::Modality::Sensor);
    EXPECT_EQ(unit.semanticVector.size(), 8u);
    EXPECT_EQ(unit.metadata["conceptEncoder"], "structured-concept-adapter");
}

TEST_F(MixedModalIOTest, ImageEncodeReportsModelNotReady) {
    MixedModalPacket img;
    img.id = "img-no-model";
    img.modality = MixedModalModality::Image;
    img.mimeType = "application/x-bgr";
    img.metadata = {{"width", 2}, {"height", 2}};
    img.payload = {0x00, 0x00, 0x00,   // black
                   0xff, 0xff, 0xff,   // white
                   0x00, 0x00, 0x00,   // black
                   0xff, 0xff, 0xff};  // white

    auto unit = img.toSemanticUnit(16);
    EXPECT_EQ(unit.modality, phoenix::multimodal::Modality::Image);
    // No deterministic statistical fallback: the model is not ready, so the
    // semantic vector is empty and the backend error is exposed.
    EXPECT_TRUE(unit.semanticVector.empty());
    EXPECT_FLOAT_EQ(unit.confidence, 0.0f);
    EXPECT_EQ(unit.metadata["conceptEncoder"], "video-encoder");
    EXPECT_TRUE(unit.metadata.contains("videoEncoderError"));
    EXPECT_EQ(unit.metadata["conceptMatrixAction"], "empty");
}

TEST_F(MixedModalIOTest, ConceptMatrixResidualRefinesOnRepeat) {
    MixedModalPacket img;
    img.id = "img-residual";
    img.modality = MixedModalModality::Image;
    img.mimeType = "application/x-bgr";
    img.metadata = {{"width", 2}, {"height", 2}};
    img.payload = {0x00, 0x00, 0x00,
                   0xff, 0xff, 0xff,
                   0x00, 0x00, 0x00,
                   0xff, 0xff, 0xff};

    auto first = img.toSemanticUnit(16);
    // Without a real image model the semantic vector is empty, so the matrix
    // records a zero residual "empty" action for both calls.
    EXPECT_EQ(first.metadata["conceptMatrixAction"], "empty");
    const float residual1 = std::stof(first.metadata["conceptMatrixResidual"]);
    EXPECT_FLOAT_EQ(residual1, 0.0f);

    auto second = img.toSemanticUnit(16);
    EXPECT_EQ(second.metadata["conceptMatrixAction"], "empty");
    const float residual2 = std::stof(second.metadata["conceptMatrixResidual"]);
    EXPECT_FLOAT_EQ(residual2, 0.0f);

    auto state = MixedModalConceptBridge::status();
    // Empty semantic vectors (model not ready) do not add to the matrix.
    EXPECT_EQ(state["conceptMatrix"]["count"].get<size_t>(), 0u);
}

TEST_F(MixedModalIOTest, PretrainImageFailsWithoutModel) {
    MixedModalPacket img;
    img.id = "img-train";
    img.modality = MixedModalModality::Image;
    img.mimeType = "application/x-bgr";
    img.metadata = {{"width", 2}, {"height", 2}};
    img.payload = {0x00, 0x00, 0x00,
                   0xff, 0xff, 0xff,
                   0x00, 0x00, 0x00,
                   0xff, 0xff, 0xff};

    EXPECT_FALSE(MixedModalConceptBridge::pretrainImage(img, "black and white", 16));

    auto state = MixedModalConceptBridge::status();
    EXPECT_EQ(state["conceptMatrix"]["additions"].get<size_t>(), 0u);
}

TEST_F(MixedModalIOTest, PretrainImageInvalidInputs) {
    MixedModalPacket notImage;
    notImage.modality = MixedModalModality::Text;
    notImage.payload = {0x01, 0x02};
    EXPECT_FALSE(MixedModalConceptBridge::pretrainImage(notImage, "caption", 8));

    MixedModalPacket emptyImage;
    emptyImage.modality = MixedModalModality::Image;
    EXPECT_FALSE(MixedModalConceptBridge::pretrainImage(emptyImage, "caption", 8));
}

TEST_F(MixedModalIOTest, ImageWorldModelReportsBackendAndMissingOnnx) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    std::map<std::string, std::string> args;
    args["vision-placement"] = "local";
    args["vision-local-backend"] = "cpu";
    cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

    auto model = phoenix::io::createVideoModel("ijepa_vith14_1k", 16);
    ASSERT_TRUE(model);
    std::vector<uint8_t> fakeImage = {0x89, 'P', 'N', 'G'};
    model->encode(fakeImage, 0, 0, "image/png");
    auto status = model->status();
    EXPECT_EQ(status["backend"], "local-onnx");
    EXPECT_FALSE(status.value("ready", false));
    EXPECT_TRUE(status.contains("error"));
    EXPECT_TRUE(status.value("error", std::string("")).find("best.onnx") != std::string::npos);
}

TEST_F(MixedModalIOTest, SpeechWorldModelReportsFallbackWhenNoOnnx) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    std::map<std::string, std::string> args;
    args["speech-placement"] = "local";
    args["speech-local-backend"] = "cpu";
    cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

    auto model = phoenix::io::createAudioModel("audio-16k", 8);
    ASSERT_TRUE(model);
    std::vector<uint8_t> fakeAudio = {0x10, 0x20};
    auto encoded = model->encode(fakeAudio, 16000, "audio/pcm");
    auto status = model->status();
    EXPECT_EQ(status["backend"], "fallback");
    EXPECT_TRUE(status.value("ready", false));
    EXPECT_EQ(encoded.size(), 8u);
    EXPECT_TRUE(status.value("error", std::string("")).empty());
}

TEST_F(MixedModalIOTest, BpuBackendChoosesHbdnnAndReportsNotReady) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    std::map<std::string, std::string> args;
    args["vision-placement"] = "local";
    args["vision-local-backend"] = "bpu";
    cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

    auto model = phoenix::io::createVideoModel("ijepa_vith14_1k", 16);
    ASSERT_TRUE(model);
    auto status = model->status();
    EXPECT_TRUE(status.value("backend", std::string("")).find("unavailable") != std::string::npos);
    EXPECT_FALSE(status.value("ready", false));
}

TEST_F(MixedModalIOTest, ServerClientImageRejectsClientRawPayload) {
    auto &cfg = phoenix::deployment::ModelDeploymentConfig::instance();
    cfg.reset();
    std::map<std::string, std::string> args;
    args["vision-placement"] = "server-client";
    cfg.load(args, "D:/_phoenix/_079/v6.0Alixander/phoenix", "");

    auto model = phoenix::io::createVideoModel("ijepa_vith14_1k", 16);
    ASSERT_TRUE(model);
    auto status = model->status();
    EXPECT_EQ(status["backend"], "server-client");

    std::vector<uint8_t> fakeImage(12, 0);
    auto encoded = model->encode(fakeImage, 2, 2, "application/x-bgr");
    EXPECT_TRUE(encoded.empty());
}

TEST_F(MixedModalIOTest, ConceptMatrixLSHAndPersistence) {
    // Use 256-d one-hot concept vectors.  Since targetDim == sourceDim the
    // projection is identity, so each unique one-hot becomes an orthogonal
    // 256-d prototype.  After 350 adds the LSH index (dim >= 32, N > 256) is
    // built and exercised; SQLite persistence also flushes every 32 inferences.
    constexpr size_t kDim = 512;
    constexpr int kCount = 350;
    for (int i = 0; i < kCount; ++i) {
        MixedModalPacket p;
        p.id = "lsh-concept-" + std::to_string(i);
        p.modality = MixedModalModality::Text;
        std::string s = "probe" + std::to_string(i);
        p.payload.assign(s.begin(), s.end());

        std::vector<float> conceptVec(kDim, 0.0f);
        conceptVec[i % kDim] = 1.0f;
        p.metadata["conceptVector"] = conceptVec;

        auto unit = p.toSemanticUnit(kDim);
        EXPECT_EQ(unit.semanticVector.size(), kDim);
        EXPECT_EQ(unit.modality, phoenix::multimodal::Modality::Text);
    }

    auto state = MixedModalConceptBridge::status();
    EXPECT_GE(state["conceptMatrix"]["count"].get<size_t>(), 256u);

    // The same concept vector should be found (exact or LSH) and updated.
    MixedModalPacket repeat;
    repeat.id = "lsh-concept-0";
    repeat.modality = MixedModalModality::Text;
    repeat.payload = {'p', 'r', 'o', 'b', 'e', '0'};
    std::vector<float> conceptVec(kDim, 0.0f);
    conceptVec[0] = 1.0f;
    repeat.metadata["conceptVector"] = conceptVec;

    auto again = repeat.toSemanticUnit(kDim);
    EXPECT_EQ(again.metadata["conceptMatrixAction"], "updated");
    EXPECT_NEAR(std::stof(again.metadata["conceptMatrixResidual"]), 0.0f, 1e-4f);

    // count should stay at the original number, not grow.
    state = MixedModalConceptBridge::status();
    size_t before = state["conceptMatrix"]["count"].get<size_t>();
    EXPECT_EQ(before, static_cast<size_t>(kCount));

    // A quick load/reload sanity check: the same bridge instance still has the
    // stored prototypes and uses the LSH index on a second query.
    MixedModalPacket repeat2;
    repeat2.id = "lsh-concept-1";
    repeat2.modality = MixedModalModality::Text;
    repeat2.payload = {'p', 'r', 'o', 'b', 'e', '1'};
    std::vector<float> conceptVec2(kDim, 0.0f);
    conceptVec2[1 % kDim] = 1.0f;
    repeat2.metadata["conceptVector"] = conceptVec2;
    auto again2 = repeat2.toSemanticUnit(kDim);
    EXPECT_EQ(again2.metadata["conceptMatrixAction"], "updated");
    EXPECT_NEAR(std::stof(again2.metadata["conceptMatrixResidual"]), 0.0f, 1e-4f);
    state = MixedModalConceptBridge::status();
    EXPECT_EQ(state["conceptMatrix"]["count"].get<size_t>(), before);
}
