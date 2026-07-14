/* transformer.hpp - Transformer neural network implementation
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.
   
   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   079 Project is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with 079 Project.  If not, see <http://www.gnu.org/licenses/>. */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <chrono>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <deque>

#include <nlohmann/json.hpp>

namespace transformer {

using json = nlohmann::json;

/* Training sample structure for supervised learning.
   Contains input text, target output, and graph context for
   graph-enhanced transformer training. */
struct TrainSample {
	std::string input;   /* Input text sequence */
	std::string target;  /* Target output sequence */
	std::string graph;   /* Graph context in textual format */
};

/* Transformer hyperparameters and configuration.
   Contains all configurable parameters for the transformer model
   including architecture, training, and generation settings. */
struct TransformerParams {
	int vocabSize{8192};              /* Vocabulary size */
	int dModel{128};                  /* Model dimension */
	int nHeads{4};                    /* Number of attention heads */
	int nLayers{2};                   /* Number of transformer layers */
	int dFF{256};                     /* Feed-forward dimension */
	int maxLen{512};                  /* Maximum sequence length */
	int maxTokens{256};               /* Maximum generation tokens */
	float lr{0.001f};                 /* Learning rate */
	bool useLrSchedule{true};         /* Enable learning rate scheduling */
	int lrWarmupSteps{200};           /* Learning rate warmup steps */
	int lrCosineSteps{2000};          /* Cosine annealing steps */
	float lrMinRatio{0.1f};           /* Minimum learning rate ratio */
	int gradAccumSteps{1};            /* Gradient accumulation steps */
	float lossNoiseAlpha{0.0f};       /* Loss noise alpha for regularization */
	int attnChunkSize{0};             /* Attention chunk size (0 = disabled) */
	float graphWeight{0.35f};          /* Graph context fusion weight */
	float verifyThreshold{0.28f};     /* Verification threshold */
	float temperature{1.0f};          /* Sampling temperature */
	bool dynamicSampling{false};      /* Enable dynamic sampling */
	float temperatureMin{0.0f};       /* Minimum temperature for dynamic sampling */
	float temperatureMax{0.0f};       /* Maximum temperature for dynamic sampling */
	int topK{0};                      /* Top-k sampling (0 = disabled) */
	float topP{0.0f};                 /* Top-p (nucleus) sampling */
	float topPMin{0.0f};             /* Minimum top-p for dynamic sampling */
	float topPMax{0.0f};             /* Maximum top-p for dynamic sampling */
	int dynamicWindow{64};            /* Dynamic sampling window size */
	float dynamicRepetitionBoost{0.25f}; /* Repetition boost for dynamic sampling */
	float labelSmoothing{0.0f};       /* Label smoothing factor */
	float gradClip{1.0f};             /* Gradient clipping threshold */
	float adamBeta1{0.9f};            /* Adam optimizer beta1 */
	float adamBeta2{0.999f};          /* Adam optimizer beta2 */
	float adamEps{1e-8f};             /* Adam optimizer epsilon */
	float weightDecay{0.0f};          /* Weight decay (L2 regularization) */
	float repetitionPenalty{1.0f};     /* Repetition penalty */
	bool repetitionSegmented{false};   /* Use segmented repetition penalty */
	int repetitionWindowRecent{32};   /* Recent repetition window */
	int repetitionWindowMid{128};     /* Mid repetition window */
	float repetitionPenaltyRecent{1.2f}; /* Recent repetition penalty */
	float repetitionPenaltyMid{1.1f};  /* Mid repetition penalty */
	float repetitionPenaltyOld{1.05f}; /* Old repetition penalty */
	int noRepeatNgram{0};            /* No-repeat n-gram size */
	int minNewTokens{0};              /* Minimum new tokens to generate */
	bool enableDiagnostics{false};   /* Enable diagnostic output */
	bool useAlibi{false};             /* Use ALiBi (Attention with Linear Biases) */
	float alibiSlope{0.8f};           /* ALiBi slope parameter */
	float bidirectionalWeight{0.0f};  /* Bidirectional attention weight */
	float selfFlowWeight{0.0f};      /* Self-flow attention weight */
	int cacheMaxEntries{256};        /* Maximum cache entries */
	int cacheTokenEntries{512};       /* Maximum token cache entries */
	int cacheTtlMs{120000};           /* Cache time-to-live in milliseconds */
	bool enableMoE{false};            /* Enable Mixture of Experts */
	int moeExperts{4};                /* Number of MoE experts */
	int moeTopK{2};                   /* Top-k experts to route to */
	float moeAuxWeight{0.01f};        /* MoE auxiliary loss weight */
	bool enableMLA{false};            /* Enable Multi-head Latent Attention */
	int mlaRank{8};                   /* MLA rank */
	float mlaScale{0.35f};            /* MLA scale factor */
	bool enableCoefAttention{false};  /* Enable coefficient attention */
	float coefAttentionScale{0.15f};  /* Coefficient attention scale */
	bool enableHierEmbedding{false};  /* Enable hierarchical embedding */
	int hierStride{16};               /* Hierarchical embedding stride */
	int windowClipTokens{0};          /* Window clip tokens */
	int draftTokens{1};               /* Number of draft tokens */
	bool enableMultiTokenObjective{true}; /* Enable multi-token objective */
	int multiTokenHeads{2};           /* Multi-token objective heads */
	float multiTokenLossWeight{0.30f}; /* Multi-token loss weight */
	bool enableCoTScaffold{false};   /* Enable Chain-of-Thought scaffold */
	int cotSteps{3};                  /* CoT steps */
	bool enableProgramSynthesis{true}; /* Enable program synthesis */
	float programSynthesisBias{0.25f}; /* Program synthesis bias */
	bool enableDynamicReasoning{true}; /* Enable dynamic reasoning */
	int simpleTokenThreshold{10};     /* Simple token threshold */
	bool enableRLHF{false};           /* Enable RLHF (Reinforcement Learning from Human Feedback) */
	float rlhfLr{0.0006f};            /* RLHF learning rate */
	int rlhfBatchSize{4};             /* RLHF batch size */
	int rlhfReplaySize{2048};         /* RLHF replay buffer size */
	float rlhfMargin{0.15f};          /* RLHF margin */
	bool enableIRL{true};             /* Enable IRL (Inverse Reinforcement Learning) */
	float irlLr{0.002f};              /* IRL learning rate */
	bool enableAddonModuleLearning{true}; /* Enable addon module learning */
	float addonRewardScale{0.35f};    /* Addon reward scale */
	std::string tokenizerMode{"bpe"}; /* Tokenizer mode: "bpe" or "hash" */
};

/* Dense matrix structure for neural network operations.
   Provides row-major storage with element access via operator(). */
struct Matrix {
	int rows{0};                     /* Number of rows */
	int cols{0};                     /* Number of columns */
	std::vector<float> data;         /* Matrix data in row-major order */

	Matrix() = default;
	Matrix(int r, int c, float v = 0.0f) : rows(r), cols(c), data((size_t)r * (size_t)c, v) {}

	/* Element access operators */
	float &operator()(int r, int c) { return data[(size_t)r * (size_t)cols + (size_t)c]; }
	float operator()(int r, int c) const { return data[(size_t)r * (size_t)cols + (size_t)c]; }
};

/* Linear (fully-connected) layer with Adam optimizer state.
   Implements y = Wx + b with momentum and Adam optimizer statistics. */
struct Linear {
	Matrix w;                      /* Weight matrix */
	std::vector<float> b;         /* Bias vector */
	std::vector<float> mW;        /* Adam first moment for weights */
	std::vector<float> vW;        /* Adam second moment for weights */
	std::vector<float> mB;        /* Adam first moment for biases */
	std::vector<float> vB;        /* Adam second moment for biases */
	std::vector<float> gW;        /* Gradient for weights */
	std::vector<float> gB;        /* Gradient for biases */
	int step{0};                  /* Training step counter */

	Linear() = default;
	Linear(int in, int out);
	std::vector<float> forward(const std::vector<float> &x) const;
};

/* Layer normalization with Adam optimizer state.
   Normalizes across the feature dimension for stable training. */
struct LayerNorm {
	std::vector<float> gamma;      /* Scale parameter */
	std::vector<float> beta;       /* Shift parameter */
	std::vector<float> mGamma;    /* Adam first moment for gamma */
	std::vector<float> vGamma;    /* Adam second moment for gamma */
	std::vector<float> mBeta;     /* Adam first moment for beta */
	std::vector<float> vBeta;     /* Adam second moment for beta */
	std::vector<float> gGamma;    /* Gradient for gamma */
	std::vector<float> gBeta;     /* Gradient for beta */
	int step{0};                  /* Training step counter */
	float eps{1e-5f};             /* Epsilon for numerical stability */

	LayerNorm() = default;
	explicit LayerNorm(int d);
	std::vector<float> forward(const std::vector<float> &x) const;
};

/* Multi-head attention mechanism with optional ALiBi, MLA, and coefficient attention.
   Splits attention into multiple heads for parallel processing of different representation subspaces. */
struct MultiHeadAttention {
	int nHeads{4};                  /* Number of attention heads */
	int dModel{128};                /* Model dimension */
	int dHead{32};                 /* Dimension per head */
	bool useAlibi{false};           /* Use ALiBi (Attention with Linear Biases) */
	float alibiSlope{0.8f};         /* ALiBi slope parameter */
	bool useCoefAttention{false};   /* Use coefficient attention */
	float coefAttentionScale{0.15f}; /* Coefficient attention scale */
	bool useMLA{false};             /* Use Multi-head Latent Attention */
	int mlaRank{8};                 /* MLA rank */
	float mlaScale{0.35f};          /* MLA scale factor */
	Linear wq;                      /* Query projection */
	Linear wk;                      /* Key projection */
	Linear wv;                      /* Value projection */
	Linear wo;                      /* Output projection */

	MultiHeadAttention() = default;
	MultiHeadAttention(int dModel, int nHeads);
	std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x,
											bool causal,
											const std::vector<std::vector<float>> *memory) const;
};

/* Feed-forward network with optional Mixture of Experts (MoE).
   Implements two-layer MLP with optional expert routing for conditional computation. */
struct FeedForward {
	Linear w1;                      /* First linear layer */
	Linear w2;                      /* Second linear layer */
	bool moeEnabled{false};         /* Enable Mixture of Experts */
	int moeTopK{2};                 /* Top-k experts to route to */
	float moeAuxWeight{0.01f};      /* MoE auxiliary loss weight */
	Linear moeGate;                 /* MoE gating network */
	std::vector<Linear> moeW1;     /* MoE expert first layers */
	std::vector<Linear> moeW2;     /* MoE expert second layers */
	mutable std::vector<float> expertUsageEma; /* Expert usage EMA for load balancing */

	FeedForward() = default;
	FeedForward(int dModel, int dFF);
	std::vector<float> forward(const std::vector<float> &x) const;
};

/* Transformer encoder layer with self-attention and feed-forward.
   Standard transformer encoder block with pre-layer normalization. */
struct EncoderLayer {
	MultiHeadAttention selfAttn;    /* Self-attention mechanism */
	FeedForward ffn;               /* Feed-forward network */
	LayerNorm ln1;                 /* First layer norm */
	LayerNorm ln2;                 /* Second layer norm */

	EncoderLayer() = default;
	EncoderLayer(int dModel, int nHeads, int dFF);
	std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x) const;
};

/* Transformer decoder layer with self-attention, cross-attention, and feed-forward.
   Standard transformer decoder block for sequence-to-sequence tasks. */
struct DecoderLayer {
	MultiHeadAttention selfAttn;    /* Self-attention mechanism */
	MultiHeadAttention crossAttn;   /* Cross-attention to encoder output */
	FeedForward ffn;               /* Feed-forward network */
	LayerNorm ln1;                 /* First layer norm */
	LayerNorm ln2;                 /* Second layer norm */
	LayerNorm ln3;                 /* Third layer norm */

	DecoderLayer() = default;
	DecoderLayer(int dModel, int nHeads, int dFF);
	std::vector<std::vector<float>> forward(const std::vector<std::vector<float>> &x,
											const std::vector<std::vector<float>> &memory) const;
};

/* Tokenizer for text encoding/decoding with BPE and hash-based modes.
   Supports both byte-pair encoding (BPE) and simple hash-based tokenization. */
class Tokenizer {
public:
	explicit Tokenizer(int vocabSize, const std::string &mode = "hash");
	void observe(const std::string &text) const; /* Observe text for BPE training */
	std::vector<int> encode(const std::string &text, int maxLen, bool addBosEos) const; /* Encode text to token IDs */
	std::string decode(const std::vector<int> &ids) const; /* Decode token IDs to text */
	int activeVocabLimit() const; /* Get active vocabulary size */
	json toJson() const; /* Serialize tokenizer state */
	bool fromJson(const json &state, std::string &error); /* Deserialize tokenizer state */
	const std::string &mode() const { return mode_; } /* Get tokenization mode */
	void setMode(const std::string &mode); /* Set tokenization mode */

private:
	int vocabSize_{8192};           /* Maximum vocabulary size */
	std::string mode_{"hash"};      /* Tokenization mode: "hash" or "bpe" */
	mutable std::mutex mu_;          /* Mutex for thread safety */
	mutable bool bpeReady_{false};  /* BPE training complete flag */
	mutable std::vector<std::string> corpus_; /* Training corpus */
	mutable size_t corpusChars_{0};  /* Total corpus characters */
	size_t corpusMaxChars_{200000}; /* Maximum corpus size for BPE training */
	int corpusMinChars_{120};       /* Minimum corpus size to trigger BPE training */
	mutable std::vector<std::string> bpeIdToToken_; /* BPE ID to token mapping */
	mutable std::unordered_map<std::string, int> bpeTokenToId_; /* BPE token to ID mapping */
	struct Merge {
		int a{0};                    /* First token in merge pair */
		int b{0};                    /* Second token in merge pair */
		int id{0};                   /* Merged token ID */
	};
	mutable std::vector<Merge> bpeMerges_; /* BPE merge operations */
	int bpeMaxMerges_{2000};        /* Maximum BPE merges */

	static std::string normalizeText(const std::string &text); /* Normalize text for tokenization */
	void ensureBpeReady() const;    /* Ensure BPE is trained if needed */
	void trainBpe() const;           /* Train BPE on corpus */
	std::vector<int> encodeBpe(const std::string &text) const; /* Encode using BPE */
};

/* Transformer model with encoder-decoder architecture and graph fusion.
   Implements a full transformer with graph context integration for enhanced reasoning. */
class TransformerModel {
public:
	explicit TransformerModel(const TransformerParams &params);
	const TransformerParams &params() const { return params_; }

	std::vector<std::vector<float>> encode(const std::vector<int> &tokens) const; /* Encode tokens to hidden states */
	std::vector<std::vector<float>> decode(const std::vector<int> &tokens,
										   const std::vector<std::vector<float>> &memory) const; /* Decode with cross-attention */
	std::vector<float> logitsAt(const std::vector<float> &hidden) const; /* Get logits from hidden state */
	std::vector<int> generate(const std::vector<int> &inputTokens,
						  const std::vector<int> &graphTokens,
					  int maxTokens,
					  float graphWeight,
					  int vocabLimit = -1) const; /* Generate with graph tokens */
	std::vector<int> generate(const std::vector<int> &inputTokens,
						  const std::vector<int> &graphTokens,
					  const std::vector<std::vector<float>> &graphEmbeddings,
					  int maxTokens,
					  float graphWeight,
					  int vocabLimit = -1) const; /* Generate with graph embeddings */
	std::vector<std::vector<float>> fuseMemory(const std::vector<std::vector<float>> &textMem,
										 const std::vector<std::vector<float>> &graphMem,
										 float graphWeight) const; /* Fuse text and graph memory */
	float trainOnSample(const std::vector<int> &inputTokens,
						const std::vector<int> &graphTokens,
						const std::vector<int> &targetTokens,
						float lr); /* Train on a single sample */

	json toJson() const; /* Serialize model */
	json stateDict() const; /* Get model state dictionary */
	bool loadStateDict(const json &state, std::string &error); /* Load model state */
	void updateParams(const TransformerParams &p); /* Update hyperparameters */

private:
	TransformerParams params_;       /* Model hyperparameters */
	Matrix tokEmbed_;                /* Token embedding matrix */
	Matrix posEmbed_;                /* Positional embedding matrix */
	Linear outProj_;                 /* Output projection layer */
	MultiHeadAttention fuseAttn_;    /* Fusion attention for graph integration */
	Linear fuseGate_;                /* Fusion gate for graph integration */
	std::vector<EncoderLayer> encLayers_; /* Encoder layers */
	std::vector<DecoderLayer> decLayers_; /* Decoder layers */
	Tokenizer tokenizer_;            /* Tokenizer instance */
	std::vector<float> outMom1_;    /* Output Adam first moment */
	std::vector<float> outMom2_;    /* Output Adam second moment */
	std::vector<float> outGradW_;   /* Output weight gradients */
	std::vector<float> outGradB_;   /* Output bias gradients */
	int outStep_{0};                /* Output training step */
	std::vector<float> tokMom1_;    /* Token embedding Adam first moment */
	std::vector<float> tokMom2_;    /* Token embedding Adam second moment */
	std::vector<float> posMom1_;    /* Position embedding Adam first moment */
	std::vector<float> posMom2_;    /* Position embedding Adam second moment */
	std::vector<float> tokGrad_;    /* Token embedding gradients */
	std::vector<float> posGrad_;    /* Position embedding gradients */
	int embedStep_{0};              /* Embedding training step */
	int trainStep_{0};              /* Total training steps */
	float lossEma_{0.0f};           /* Exponential moving average of loss */

	std::vector<std::vector<float>> embedTokens(const std::vector<int> &tokens) const; /* Embed tokens */
};

/* High-level transformer service with caching, RLHF, and feedback learning.
   Provides chat interface, training, optimization, and human feedback integration. */
class TransformerService {
public:
	TransformerService();

	json chat(const std::string &text, const std::string &graphContext, int maxTokens); /* Chat with graph context (text) */
	json chat(const std::string &text, const std::string &graphContext,
			  const std::vector<std::vector<float>> &graphEmbeddings, int maxTokens); /* Chat with graph embeddings */
	json pretrain(const std::vector<TrainSample> &samples, int epochs, float lr); /* Pretrain on samples */
	json jointTrain(const std::vector<TrainSample> &samples, int epochs, float lr, float graphWeight); /* Joint training with graph */
	json optimizeGA(const std::vector<TrainSample> &samples, int generations, int population); /* Genetic algorithm optimization */
	json verify(const std::string &text, const std::string &graphContext, const std::string &reply) const; /* Verify reply consistency */
	void addFeedback(const std::string &reply, float score); /* Add scalar feedback */
	void addPreferenceFeedback(const std::string &text,
				  const std::string &graphContext,
				  const std::string &chosen,
				  const std::string &rejected,
				  float reward,
				  const std::string &source); /* Add preference feedback for RLHF */
	json trainFromFeedback(int steps); /* Train from collected feedback */
	json rlhfStats() const; /* Get RLHF statistics */
	json params() const; /* Get current parameters */
	void applyParams(const json &patch); /* Apply parameter patch */
	json saveCheckpoint(const std::string &path) const; /* Save model checkpoint */
	json loadCheckpoint(const std::string &path); /* Load model checkpoint */

private:
	TransformerParams params_;       /* Service hyperparameters */
	TransformerModel model_;        /* Transformer model instance */
	Tokenizer tokenizer_;            /* Tokenizer instance */
	std::mt19937 rng_{0xC0FFEEu};  /* Random number generator */
	mutable std::mutex cacheMu_;    /* Cache mutex */
	std::list<std::string> replyOrder_; /* LRU order for reply cache */
	std::unordered_map<std::string, std::pair<json, std::pair<int64_t, std::list<std::string>::iterator>>> replyCache_; /* Reply cache */
	std::list<std::string> tokenOrder_; /* LRU order for token cache */
	std::unordered_map<std::string, std::pair<std::vector<int>, std::list<std::string>::iterator>> tokenCache_; /* Token cache */
	mutable std::mutex rlhfMu_;    /* RLHF mutex */
	std::unordered_map<std::string, float> rlhfReplyScore_; /* Reply scores for RLHF */
	std::vector<float> irlWeights_; /* Inverse reinforcement learning weights */
	std::unordered_map<std::string, float> addonSkill_; /* Addon skill scores */
	struct PreferenceSample
	{
		std::string text;            /* Input text */
		std::string graph;           /* Graph context */
		std::string chosen;          /* Chosen reply */
		std::string rejected;        /* Rejected reply */
		float reward{0.0f};          /* Reward value */
		std::string source;          /* Feedback source */
	};
	std::deque<PreferenceSample> prefReplay_; /* Preference replay buffer */
 	int64_t feedbackTrainSteps_{0}; /* Feedback training step counter */

	float sampleLoss(const TrainSample &s); /* Compute loss for a sample */
	float jaccard(const std::vector<int> &a, const std::vector<int> &b) const; /* Jaccard similarity */
	float computeContrastiveLoss(const std::vector<int> &a, const std::vector<int> &b) const; /* Contrastive loss */
};

} // namespace transformer
