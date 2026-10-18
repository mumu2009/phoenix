#pragma once
inline int selfModValue() { return 2; }
inline int selfModClamp(int x) { return (x < 0) ? 0 : x; }
inline int selfModSum(int n) { return n < 0 ? 0 : n*(n+1)/2; }
