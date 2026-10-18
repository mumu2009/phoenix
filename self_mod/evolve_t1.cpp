std::string evolveCompute() {
uint32_t state = 20261018u; for (int i = 0; i < 10000; i++) { state = state * 1103515245u + 12345u; } return std::to_string(state); }
