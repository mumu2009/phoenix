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

   Copyright (C) 2026 079 Project */

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

}  // namespace v7
}  // namespace phoenix

#endif  // PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP
