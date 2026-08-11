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
   autoregressive token loop inside llama-server: the client only sees the
   /apply-template boundary, /phx/enc as the tokenizer, /phx/generate as the
   inference engine, and the returned text as the detokenizer output.  This
   matches the standard multimodal design where the tokenizer/detokenizer
   live at the I/O boundary and the model is token-in-token-out.  The
   /phx/enc and /phx/dec endpoints remain exposed for debugging and for
   modality-specific decoders (audio/video) that operate on the unit-query
   stream returned by /phx/generate.

   Copyright (C) 2026 079 Project */

#ifndef PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP
#define PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP

#include <string>

#include <nlohmann/json.hpp>

namespace phoenix {
namespace v7 {

// Performs a full chat generation turn against a patched llama-server
// exposing /apply-template, /phx/enc, /phx/infer, /phx/dec and
// /detokenize, using hidden-state (unit) I/O exclusively for the
// generation loop rather than the text-based /api/chat endpoint.
//
// Returns a JSON object with keys: ok (bool), reply (string),
// model (string), provider (string) == "llamacpp", and on failure
// an additional "error" (string) key.
nlohmann::json llamaSplitChat(const std::string &baseUrl, int timeoutMs,
                               const std::string &model,
                               const std::string &text,
                               const std::string &graphContext, int maxTokens,
                               const nlohmann::json &inferenceOptions);

// Same request/response contract as llamaSplitChat(), but talks only to
// llama-server's native, unmodified /apply-template + /v1/chat/completions
// endpoints (plain text I/O) and never touches /phx/enc or /phx/infer.
// Use this when main.inference.llamaUseSplitBackend is disabled. The
// llama_server_mods/enc_dec_separation.patch has been fixed so the split
// endpoints no longer crash; this path remains the fallback when the split
// backend is disabled or unavailable.
nlohmann::json llamaTextOnlyChat(const std::string &baseUrl, int timeoutMs,
                                  const std::string &model,
                                  const std::string &text,
                                  const std::string &graphContext,
                                  int maxTokens,
                                  const nlohmann::json &inferenceOptions);

}  // namespace v7
}  // namespace phoenix

#endif  // PHOENIX_LLAMA_SPLIT_BACKEND_CLIENT_HPP
