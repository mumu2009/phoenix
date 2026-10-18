std::string evolveCompute() {
uint32_t state = 20261018u;  for (int i = 0; i < 250000; i++) {  state = state * 1664525u + 1013904223u;  }  return std::to_string(state); }
