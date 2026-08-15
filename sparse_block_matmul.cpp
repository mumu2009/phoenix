/* sparse_block_matmul.cpp - Block-sparse matmul implementation
   Copyright (C) 2026 079 Project */

#include "sparse_block_matmul.hpp"

#include <algorithm>
#include <cmath>

namespace phoenix {
namespace math {

void denseMatMul(const float *A, const float *B, float *C,
                 size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
    }
  }
}

size_t blockSparseMatMul(const float *A, const float *B, float *C,
                         size_t M, size_t N, size_t K,
                         size_t blockRows, float threshold) {
  if (blockRows == 0) {
    blockRows = 32;
  }
  size_t skippedBlocks = 0;
  for (size_t i0 = 0; i0 < M; i0 += blockRows) {
    const size_t iEnd = std::min(M, i0 + blockRows);
    bool skip = true;
    for (size_t i = i0; i < iEnd && skip; ++i) {
      for (size_t k = 0; k < K; ++k) {
        if (std::fabs(A[i * K + k]) >= threshold) {
          skip = false;
          break;
        }
      }
    }
    if (skip) {
      ++skippedBlocks;  // C rows [i0, iEnd) remain zero.
      continue;
    }
    for (size_t i = i0; i < iEnd; ++i) {
      for (size_t j = 0; j < N; ++j) {
        float acc = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          acc += A[i * K + k] * B[k * N + j];
        }
        C[i * N + j] = acc;
      }
    }
  }
  return skippedBlocks;
}

double frobeniusErrorBound(size_t skippedEntries, float threshold,
                           double maxColNormB) {
  // Each skipped row r contributes error |A_r . B_col| <= ||A_r||_2 * ||B_col||_2
  // <= threshold * sqrt(K) * ||B_col||_2 by Cauchy-Schwarz; aggregated:
  return static_cast<double>(threshold) *
         std::sqrt(static_cast<double>(skippedEntries)) * maxColNormB;
}

}  // namespace math
}  // namespace phoenix
