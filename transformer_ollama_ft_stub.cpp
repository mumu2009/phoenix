/* transformer_ollama_ft_stub.cpp - Stub for Cython-generated ollama fine-tuning
   bridge functions that cannot compile on RDK X5 (requires Python314 headers).
   All functions return failure with an error message.

   Copyright (C) 2026 079 Project */

#include <cstring>
#include <string>

extern "C" {

static void setError(char *error_buf, size_t error_cap, const char *msg) {
    if (error_buf && error_cap > 0) {
        std::string err = msg;
        size_t n = std::min(err.size(), error_cap - 1);
        std::memcpy(error_buf, err.c_str(), n);
        error_buf[n] = '\0';
    }
}

int transformer_ollama_ft_bridge_set_module(const char * /*module_name*/,
                                            char *error_buf, size_t error_cap) {
    setError(error_buf, error_cap,
             "transformer_ollama_ft_bridge not available on RDK X5 build");
    return 0;
}

int transformer_ollama_ft_bridge_init(char *error_buf, size_t error_cap) {
    setError(error_buf, error_cap,
             "transformer_ollama_ft_bridge not available on RDK X5 build");
    return 0;
}

int transformer_ollama_ft_bridge_run_main(int /*argc*/,
                                          const char *const * /*argv*/,
                                          char *error_buf, size_t error_cap) {
    setError(error_buf, error_cap,
             "transformer_ollama_ft_bridge not available on RDK X5 build");
    return 0;
}

int transformer_ollama_ft_bridge_train(
    const char * /*append_corpus*/, const char * /*output_dir*/,
    const char * /*ollama_model*/, const char * /*hf_model*/,
    int /*self_play_pairs*/, int /*epochs*/, double /*lr*/,
    char *error_buf, size_t error_cap) {
    setError(error_buf, error_cap,
             "transformer_ollama_ft_bridge not available on RDK X5 build");
    return 0;
}

void transformer_ollama_ft_bridge_finalize(void) {
    /* no-op */
}

}  // extern "C"
