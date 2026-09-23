/* llama_split_backend_client.hpp - Phoenix client for the patched
   llama-server's split unit-level endpoints (/phx/enc, /phx/infer,
   /phx/dec).

   Conceptually, the split backend is unit-query-in, unit-query-out:
     /phx/enc  : text/audio/video -> unit query (hidden state)
     /phx/infer: unit query -> unit query
     /phx/dec  : unit query -> text/audio/video
   The output of /phx/dec is delivered to the user/AsyncLearning and is not
   fed directly back into /phx/infer.

   The client uses the server-side /phx/generate endpoint to keep the
   autoregressive token loop inside llama-server: raw causal text goes to
   /phx/enc, /phx/generate runs enc-infer-dec, and the returned text is
   the detokenizer output. No ChatML / apply-template / assistant role.
   The /phx/enc and /phx/dec endpoints remain exposed for debugging and for
   modality-specific decoders (audio/video) that operate on the unit-query
   stream returned by /phx/generate.
 */

#ifndef PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP
#define PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP

#include <string>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace v7 {

// Performs one enc-infer-dec completion against a patched llama-server
// exposing /phx/enc, /phx/infer, /phx/dec and /phx/generate.
//
// Returns a JSON object with keys: ok (bool), reply (string),
// model (string), provider (string) == "llamacpp", and on failure
// an additional "error" (string) key.
nlohmann::json llamaSplitChat(const std::string &baseUrl, int timeoutMs,
                               const std::string &model,
                               const std::string &text,
                               const std::string &graphContext, int maxTokens,
                               const nlohmann::json &inferenceOptions);

/* Floor for /phx/generate and /completion n_predict. Tiny values (≤8)
   have tripped GGML_ASSERT(result_output) on the patched server. */
inline int minPredictTokens() { return 16; }

/* Short-chat HTTP budget aligned with n_predict, not a 12s knife.
   RDK decode is ~1.5–2s/token so 16 tokens need 24s+ plus prefill.
   Floor 90s (board first token <60s then finish); cap 120s (<<240s
   and never the 90 min mission timeout). Host usually returns much
   earlier. */
inline int shortGenerateTimeoutMs(int nPredict) {
  const int n = nPredict < minPredictTokens() ? minPredictTokens() : nPredict;
  int need = 30000 + n * 2000;
  if (need < 90000)
    need = 90000;
  if (need > 120000)
    need = 120000;
  return need;
}

inline int jsonOptInt(const nlohmann::json &opts, const char *key,
                      int fallback = 0) {
  if (!opts.is_object() || !opts.contains(key))
    return fallback;
  const auto &v = opts[key];
  if (v.is_number_integer())
    return v.get<int>();
  if (v.is_number())
    return static_cast<int>(v.get<double>());
  return fallback;
}

/* Keep the caller HTTP budget (minutes on RDK, longer on host).
   The 90–120s short-ask cap applies only to the user ≤32-token path.
   Long generate if ANY of:
     - maxTokens > 32
     - deliberateMaxTokens >= 64
     - autonomy == true
   Do NOT use `lowPriority && maxTokens>256`: default
   mission.deliberateMaxTokens=256 fails that test and gets knifed.
   lowPriority alone is not enough either — board August may tag
   interactive chat as lowPriority. */
inline bool keepCallerGenerateBudget(int maxTokens,
                                     const nlohmann::json &opts) {
  if (maxTokens > 32)
    return true;
  if (!opts.is_object())
    return false;
  if (jsonOptInt(opts, "deliberateMaxTokens") >= 64)
    return true;
  if (opts.value("autonomy", false))
    return true;
  return false;
}

/* Short ask: raise a tiny caller value to 90s, then cap at 120s.
   Long generate: return the caller budget unchanged (min 1s). */
inline int resolveGenerateTimeoutMs(int callerTimeoutMs, int maxTokens,
                                    const nlohmann::json &opts) {
  int callTimeoutMs = callerTimeoutMs < 1000 ? 1000 : callerTimeoutMs;
  if (keepCallerGenerateBudget(maxTokens, opts))
    return callTimeoutMs;
  const int aligned = shortGenerateTimeoutMs(maxTokens);
  if (callTimeoutMs < aligned)
    callTimeoutMs = aligned;
  if (callTimeoutMs > 120000)
    callTimeoutMs = 120000;
  return callTimeoutMs;
}

/* Ask llama to drop an in-flight /phx/generate. Never /health, never RST.
   Missing endpoint is a no-op. Fire-and-forget; does not wait on generate. */
void cancelPhxGenerate(const std::string &baseUrl);

}  // namespace v7
}  // namespace phoenix

#endif  // PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP
