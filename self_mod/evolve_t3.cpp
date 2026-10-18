std::string evolveCompute() {
std::string a="0", b="1"; for (int n = 2; n <= 100; ++n) { auto c = addDecimal(a, b); a = b; b = c; } return b; }
