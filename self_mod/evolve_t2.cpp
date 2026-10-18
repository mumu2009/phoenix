std::string evolveCompute() {
  int count = 0;
  for (int n = 2; n < 10000; ++n) {
    int prime = 1;
    for (int d = 2; d * d <= n; ++d) {
      if (n % d == 0) prime = 0;
    }
    if (prime == 1) count++
;
  }
  return std::to_string(count);
}
