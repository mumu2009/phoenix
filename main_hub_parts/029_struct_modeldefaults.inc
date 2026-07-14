/* Auto-generated from main.cpp
   Segment: struct_ModelDefaults
   Original lines: 3465-3487 */

#include "../phoenix_config.hpp"

struct ModelDefaults {
  double decayFactor;
  int maxMemeWords;
  int minOverlapThreshold;
  int memeNgramMin;
  int memeNgramMax;
  double maliciousThreshold;
  int learningIterations;
  int iteration;
  double threshold;
  double decay;
  double decayK;
  int maxLen;
  double edgeWeight;
  std::string activationType;
  std::string transferType;
  std::string activationCustom;
  std::string transferCustom;
  int mappingDepth;
  int reflectionTopMemes;
  int reflectionTopWords;
  double reflectionMinScore;
};

inline ModelDefaults loadModelDefaultsFromConfig() {
  ModelDefaults d;
  d.decayFactor = phoenix::cfg<double>("model_defaults.decayFactor");
  d.maxMemeWords = phoenix::cfg<int>("model_defaults.maxMemeWords");
  d.minOverlapThreshold = phoenix::cfg<int>("model_defaults.minOverlapThreshold");
  d.memeNgramMin = phoenix::cfg<int>("model_defaults.memeNgramMin");
  d.memeNgramMax = phoenix::cfg<int>("model_defaults.memeNgramMax");
  d.maliciousThreshold = phoenix::cfg<double>("model_defaults.maliciousThreshold");
  d.learningIterations = phoenix::cfg<int>("model_defaults.learningIterations");
  d.iteration = phoenix::cfg<int>("model_defaults.iteration");
  d.threshold = phoenix::cfg<double>("model_defaults.threshold");
  d.decay = phoenix::cfg<double>("model_defaults.decay");
  d.decayK = phoenix::cfg<double>("model_defaults.decayK");
  d.maxLen = phoenix::cfg<int>("model_defaults.maxLen");
  d.edgeWeight = phoenix::cfg<double>("model_defaults.edgeWeight");
  d.activationType = phoenix::cfg<std::string>("model_defaults.activationType");
  d.transferType = phoenix::cfg<std::string>("model_defaults.transferType");
  d.activationCustom = phoenix::cfg<std::string>("model_defaults.activationCustom");
  d.transferCustom = phoenix::cfg<std::string>("model_defaults.transferCustom");
  d.mappingDepth = phoenix::cfg<int>("model_defaults.mappingDepth");
  d.reflectionTopMemes = phoenix::cfg<int>("model_defaults.reflectionTopMemes");
  d.reflectionTopWords = phoenix::cfg<int>("model_defaults.reflectionTopWords");
  d.reflectionMinScore = phoenix::cfg<double>("model_defaults.reflectionMinScore");
  return d;
}
