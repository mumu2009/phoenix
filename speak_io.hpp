/* speak_io.hpp - Speech I/O interface for audio analysis and synthesis */

#ifndef SPEAK_IO_HPP
#define SPEAK_IO_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <json/json.h>

/* Speech I/O interface for audio analysis and synthesis */
class SpeakIO {
public:
    /* Audio data structure */
    struct AudioData {
        int sampleRate{16000};             /* Sample rate in Hz */
        int channels{1};                   /* Number of channels */
        std::vector<float> mono;           /* Mono audio samples */
    };

    SpeakIO();

    /* Analyze WAV bytes */
    Json::Value analyzeWavBytes(const std::vector<uint8_t> &bytes);
    /* Analyze PCM audio */
    Json::Value analyzePcm(const AudioData &audio);
    /* Synthesize text to speech */
    Json::Value synthesizeText(const std::string &text, int sampleRate = 16000, float speed = 1.0f, float pitch = 1.0f);

private:
    /* Analyze audio environment */
    Json::Value analyzeEnvironment(const AudioData &audio);
    /* Analyze audio tone */
    Json::Value analyzeTone(const AudioData &audio);
    /* Recognize speech from audio */
    std::string recognizeSpeech(const AudioData &audio, Json::Value &asrMeta);
    /* Separate audio tracks */
    Json::Value separateTracks(const AudioData &audio);

    /* Parse WAV bytes to audio data */
    bool parseWav(const std::vector<uint8_t> &bytes, AudioData &out, std::string &err);
    /* Compute RMS of audio */
    static float computeRms(const std::vector<float> &x);
    /* Compute zero-crossing rate */
    static float computeZcr(const std::vector<float> &x);
    /* Compute spectral centroid */
    static float computeSpectralCentroid(const std::vector<float> &x, int sampleRate);
    /* Estimate pitch of audio */
    static float estimatePitch(const std::vector<float> &x, int sampleRate);
    /* Classify environment from audio features */
    static std::string classifyEnvironment(float rms, float zcr, float centroid);
    /* Classify emotion from audio features */
    static std::string classifyEmotion(float rms, float pitch);
    /* Encode audio to WAV PCM16 */
    static std::vector<uint8_t> encodeWavPcm16(const AudioData &audio);
    /* Base64 encode data */
    static std::string base64Encode(const std::vector<uint8_t> &data);
};

#endif // SPEAK_IO_HPP
