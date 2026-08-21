# llama_server_mods

This directory tracks Phoenix's local modifications to the vendored
`outsides/llamacpp` checkout as **patch files**, plus the scripts used to
(re-)apply them and rebuild `llama-server` / `llama-cli`.

## Why patches instead of editing `outsides/llamacpp` directly

`outsides/` is listed in Phoenix's `.gitignore` -- it holds vendored
third-party checkouts (llama.cpp, BitNet, bullet3, etc.) that are cloned
separately rather than committed into this repo. That means:

- Any change made directly inside `outsides/llamacpp` is **not tracked by
  git** and will silently disappear if that checkout is re-cloned, wiped,
  or replaced by a fresh `git clone` of upstream llama.cpp.
- To make Phoenix's llama.cpp changes reproducible and reviewable, we keep
  them here, under version control, as ordinary `git diff`-format patch
  files, and apply them programmatically to whatever `outsides/llamacpp`
  checkout happens to exist locally.

## Files

| File | Purpose |
|---|---|
| `existing_mods.patch` | Pre-existing Phoenix modifications: an intrusive "style adapter" that biases the attention mask with an online gradient-style update (`llama_style_adapter_*` API). Unrelated to the enc/infer/dec split; kept as-is. |
| `enc_dec_separation.patch` | New: splits llama3.1-8b's forward pass into three independently callable conceptual stages (see below). |
| `apply_patches.bat` | Windows: resets `outsides/llamacpp` if needed and applies patches. Idempotent. |
| `apply_patches.sh` | Linux/RDK counterpart of `apply_patches.bat` (used by `tools/rdk_llama_setup.sh`). |
| `build_llama_server.bat` | Builds `llama`, `llama-server`, and `llama-cli` from `outsides/llamacpp/build-gcc` via `ninja` (configuring with `cmake -G Ninja` first if that directory hasn't been configured yet). Skips the actual ninja invocation if nothing under `include/`, `src/`, `examples/server/`, or these `.patch` files has changed since the last successful build (tracked via a stamp file, `build-gcc/.phoenix_llama_server_build_stamp`). |

## How to apply and build

```bat
REM from the phoenix\ directory
llama_server_mods\apply_patches.bat
llama_server_mods\build_llama_server.bat
```

Or let `compile.bat` do it automatically (see "Integration with compile.bat"
below) -- both scripts are invoked as best-effort steps before Phoenix's own
`phoenix_main.exe` is compiled.

`apply_patches.bat`:
1. Checks whether the patch stack is already applied (via
   `git apply --reverse --check`) and exits immediately if so.
2. Otherwise resets `outsides/llamacpp` to a clean checkout
   (`git reset --hard` + `git clean`) and applies both patches in order
   with `git apply`.

`build_llama_server.bat`:
1. Configures `outsides/llamacpp/build-gcc` with `cmake -G Ninja` if it has
   no `build.ninja` yet.
2. Compares file timestamps to decide whether a rebuild is needed; pass
   `--force` (or set `PHOENIX_LLAMA_SERVER_FORCE_REBUILD=1`) to always
   rebuild.
3. Runs `ninja llama llama-server llama-cli` inside `build-gcc`.
4. Writes the stamp file on success.

Both scripts print clear `[ERROR]` / `[WARN]` lines and use normal batch
exit codes (`0` success, `1` failure) so callers can decide how to react.

## What the enc / infer / dec split does

`enc_dec_separation.patch` adds a small C API on top of llama.cpp's existing
`llama_decode()` machinery (see `include/llama.h`) that lets a caller run
what `llama_decode()` normally does in one fused pass as three separate
steps, specifically for Llama-3.1-style models (`LLM_ARCH_LLAMA`, handled by
`build_llama()` in `src/llama.cpp`):

| Stage | Signature | What it does |
|---|---|---|
| **enc** | `llama_phx_encode(ctx, tokens, n_tokens, out_hidden)` | input token ids -> embeddings. Pure embedding-table lookup (`ggml_get_rows` against `model.tok_embd`, plus any active LoRA embedding deltas). Does not touch the KV cache. |
| **infer** | `llama_phx_infer(ctx, in_hidden, n_tokens, positions, out_hidden)` | hidden states -> hidden states. Runs the *entire* transformer body (all attention + FFN layers, RoPE, KV cache updates) starting from caller-supplied embeddings instead of a token lookup, and stops right before the final norm + output projection. |
| **dec** | `llama_phx_decode(ctx, in_hidden, n_tokens, out_logits)` | hidden states -> logits. Runs only the final RMSNorm + `lm_head` projection (`model.output_norm`, `model.output`). No attention, no KV cache. |

Conceptually, the split pipeline is **unit-query-in, unit-query-out**:
`enc` converts an input modality (text/audio/video) into a unit query
(hidden state), `infer` processes unit query -> unit query, and `dec`
converts the final unit query into an output modality.  Text/audio/video
only appear before `enc` or after `dec`.  The output of `dec` is delivered
to the user/consumer and also to `AsyncLearning` for gap detection and
asynchronous matrix updates; it is **not** fed back into `infer`.

For the current 8B text model, which is autoregressive and token-based,
the token loop is exposed through the server-side `POST /phx/generate`
endpoint, which encodes the prompt, runs `infer`, `dec`, sampling,
re-encodes the sampled token, and `infer`s again until the EOS token or
`max_tokens` is reached.  This keeps the text-token stepping stone inside
`llama-server`; the client only sees the apply-template boundary and the
final text output.  `enc`, `infer` and `dec` remain available as separate
endpoints for debugging and for modality-specific decoders that operate on
the unit-query stream returned by `/phx/generate`.

Implementation notes (see the patch for details):

- A new `enum llama_phx_stage { LLAMA_PHX_STAGE_FULL, LLAMA_PHX_STAGE_ENC,
  LLAMA_PHX_STAGE_INFER, LLAMA_PHX_STAGE_DEC }` plus `llama_phx_set_stage()`
  / `llama_phx_get_stage()` select which stage `build_llama()` builds a
  graph for. `LLAMA_PHX_STAGE_FULL` (the default, value `0`) reproduces the
  original, unmodified behavior exactly.
- `build_llama()` in `src/llama.cpp` gains three short-circuits:
  - `DEC`: computes `inpL` via the existing `llm_build_inp_embd()` helper
    (which already supports feeding raw embeddings in via
    `llama_batch.embd` instead of `llama_batch.token`), then jumps straight
    to norm + `lm_head`, skipping the attention/FFN stack and `inp_pos`
    / `inp_KQ_mask` construction entirely.
  - `ENC`: computes `inpL` the normal way (token embedding lookup) and
    returns immediately, before the transformer stack.
  - `INFER`: runs the transformer stack as usual (embeddings supplied via
    `llama_batch.embd`), then returns immediately after the last layer,
    before the final norm + `lm_head`.
- The `enc` and `infer` outputs are both hidden-state tensors, so they are
  named `"result_embd_pooled"` and extracted through
  `llama_decode_impl()`'s *existing*, unmodified embeddings-extraction code
  path (the one normally used for `cparams.pooling_type ==
  LLAMA_POOLING_TYPE_NONE`). The three `llama_phx_*` wrapper functions in
  `src/llama-context.cpp` temporarily flip `cparams.embeddings` /
  `cparams.pooling_type` around the underlying `llama_decode()` call and
  restore them afterwards, so this is invisible to any other code using the
  same `llama_context`.
- The `dec` output keeps the normal `"result_output"` tensor name, so it
  reuses the existing logits-extraction path unchanged.
- `struct llama_context` gains one new member, `phx_split.stage`
  (`src/llama-context.h`), defaulted to `LLAMA_PHX_STAGE_FULL` so contexts
  that never call `llama_phx_*` behave exactly as before.

### HTTP endpoints (`examples/server/server.cpp`)

Four endpoints wrap the C API for `llama-server`:

- `POST /phx/enc` -- body `{"tokens": [1,2,3]}` or `{"content": "..."}` ->
  `{"n_tokens", "n_embd", "tokens", "hidden": [[...], ...]}`.
- `POST /phx/infer` -- body `{"hidden": [[...], ...], "positions": [...]?}`
  -> `{"n_tokens", "n_embd", "hidden": [[...], ...]}`.
- `POST /phx/dec` -- body `{"hidden": [[...], ...], "with_logits": false}` ->
  `{"n_tokens", "n_vocab", "tokens": [{"token": id, "piece": "..."}, ...]}`
  (optionally including the full `"logits"` matrix when `with_logits` is
  `true`).
- `POST /phx/generate` -- body `{"content": "...", "max_tokens": N,
  "temperature": 0.0, "top_p": 0.95, "decode_text": true,
  "return_hidden": false}` -> `{"n_tokens", "tokens", "text", "hidden?"}`.
  Runs the autoregressive token-in-token-out loop internally using
  `enc/infer/dec`, and returns the generated token ids plus optional
  detokenized text and per-token hidden states.

These are implemented as direct synchronous calls into the server's
`llama_context` (guarded by a new `server_context::phx_mutex`), bypassing
the normal slot/task-queue machinery used by `/completion` and friends.
The Phoenix client now uses `/phx/generate` for text generation, so it no
longer performs the token-by-token `enc/infer/dec` loop itself.  The split
endpoints remain exposed for offline pipeline experimentation (e.g. running
`enc` on one machine, `infer` on another, and a modality-specific `dec` on
a third), but they should not be mixed with heavy concurrent `/completion`
traffic against the same server process.

## Integration with `compile.bat`

`compile.bat` calls `llama_server_mods\apply_patches.bat` and
`llama_server_mods\build_llama_server.bat` as best-effort steps before
compiling `phoenix_main.exe`. Failures there print a `[WARN]` and do **not**
fail the overall Phoenix build, unless `PHOENIX_REQUIRE_LLAMA_SERVER=1` is
set in the environment, in which case a failure in either step aborts
`compile.bat` with a non-zero exit code. See the `[STEP] Rebuild
llama-server (enc/infer/dec split)` section near the top of `compile.bat`.
