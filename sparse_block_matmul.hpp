/* sparse_block_matmul.hpp - Block-sparse matmul with zero-skip
   Copyright (C) 2026 079 Project

   This file is part of 079 Project.

   079 Project is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   Reference implementation of the "skip multiply-by-zero" optimisation for
   weight-pruned matrices (see tools/llama_prune_analyzer.py and
   doc/v7.0/llamacpp_optimization.md).

   Design decision - BLOCK granularity, not element granularity:
     - element-wise zero checks break SIMD vectorisation (every multiply
       would need a branch) and cause branch mispredictions;
     - a block check costs O(blockRows*K) comparisons, amortised over
       blockRows*K*N MACs, i.e. 1/N per MAC - negligible for large N;
     - skipping a block removes blockRows*N MACs wholesale.

   Correctness:
     - threshold = 0 (only true zeros skipped) is EXACT: identical results
       to the dense matmul;
     - threshold > 0 is approximate with the Frobenius error bound
         ||C - C'||_F <= threshold * sqrt(skippedEntries) * max_j ||B_*j||_2
       (Cauchy-Schwarz on each skipped row-block; see the .cpp).
*/
#pragma once

#include <cstddef>

namespace phoenix {
namespace math {

/** Dense reference matmul: C = A * B, A is MxK, B is KxN (row-major). */
void denseMatMul(const float *A, const float *B, float *C,
                 size_t M, size_t N, size_t K);

/**
 * @brief Block-sparse matmul.  Skips whole row-blocks of A whose |entries|
 *        are all below p threshold (C rows for such blocks stay zero).
 *
 * @return number of skipped blocks (diagnostics).
 */
size_t blockSparseMatMul(const float *A, const float *B, float *C,
                         size_t M, size_t N, size_t K,
                         size_t blockRows = 32, float threshold = 0.0f);

/** Frobenius error bound for threshold-based skipping (documentation helper). */
double frobeniusErrorBound(size_t skippedEntries, float threshold,
                           double maxColNormB);

}  // namespace math
}  // namespace phoenix
