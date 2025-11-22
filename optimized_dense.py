"""
Optimized Dense Matrix Multiplication Implementations
Task 2 - Big Data Course
Author: Yaín René Estrada Domínguez
"""

import numpy as np
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple


@dataclass
class DenseResult:
    """Results for dense matrix multiplication"""
    algorithm: str
    matrix_size: int
    execution_time: float
    gflops: float
    
    def to_dict(self):
        return asdict(self)


class OptimizedDenseMultiplication:
    """
    Collection of optimized dense matrix multiplication algorithms
    """
    
    def __init__(self):
        self.results = []
    
    @staticmethod
    def naive_multiplication(A, B):
        """
        Naive O(n^3) matrix multiplication
        Reference implementation from Task 1
        """
        n = A.shape[0]
        C = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    C[i][j] += A[i][k] * B[k][j]
        
        return C
    
    @staticmethod
    def numpy_multiplication(A, B):
        """
        NumPy optimized multiplication (uses BLAS)
        """
        return np.dot(A, B)
    
    @staticmethod
    def strassen_multiplication(A, B):
        """
        Strassen's algorithm: O(n^2.807)
        Recursive divide-and-conquer approach
        """
        n = A.shape[0]
        
        # Base case: use numpy for small matrices
        if n <= 64:
            return np.dot(A, B)
        
        # Ensure matrix size is power of 2
        if n % 2 != 0:
            # Pad to next power of 2
            next_pow2 = 2 ** int(np.ceil(np.log2(n)))
            A_padded = np.zeros((next_pow2, next_pow2))
            B_padded = np.zeros((next_pow2, next_pow2))
            A_padded[:n, :n] = A
            B_padded[:n, :n] = B
            
            result = OptimizedDenseMultiplication.strassen_multiplication(A_padded, B_padded)
            return result[:n, :n]
        
        # Divide matrices into quadrants
        mid = n // 2
        
        A11, A12 = A[:mid, :mid], A[:mid, mid:]
        A21, A22 = A[mid:, :mid], A[mid:, mid:]
        
        B11, B12 = B[:mid, :mid], B[:mid, mid:]
        B21, B22 = B[mid:, :mid], B[mid:, mid:]
        
        # Compute the 7 Strassen products
        M1 = OptimizedDenseMultiplication.strassen_multiplication(A11 + A22, B11 + B22)
        M2 = OptimizedDenseMultiplication.strassen_multiplication(A21 + A22, B11)
        M3 = OptimizedDenseMultiplication.strassen_multiplication(A11, B12 - B22)
        M4 = OptimizedDenseMultiplication.strassen_multiplication(A22, B21 - B11)
        M5 = OptimizedDenseMultiplication.strassen_multiplication(A11 + A12, B22)
        M6 = OptimizedDenseMultiplication.strassen_multiplication(A21 - A11, B11 + B12)
        M7 = OptimizedDenseMultiplication.strassen_multiplication(A12 - A22, B21 + B22)
        
        # Compute result quadrants
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        
        # Combine quadrants
        C = np.zeros((n, n))
        C[:mid, :mid] = C11
        C[:mid, mid:] = C12
        C[mid:, :mid] = C21
        C[mid:, mid:] = C22
        
        return C
    
    @staticmethod
    def blocked_multiplication(A, B, block_size=64):
        """
        Cache-optimized blocked (tiled) matrix multiplication
        Improves cache locality by processing blocks
        
        Args:
            A, B: input matrices
            block_size: size of cache blocks (typically 32-128)
        """
        n = A.shape[0]
        C = np.zeros((n, n))
        
        # Iterate over blocks
        for i in range(0, n, block_size):
            for j in range(0, n, block_size):
                for k in range(0, n, block_size):
                    # Define block boundaries
                    i_end = min(i + block_size, n)
                    j_end = min(j + block_size, n)
                    k_end = min(k + block_size, n)
                    
                    # Multiply blocks
                    C[i:i_end, j:j_end] += np.dot(
                        A[i:i_end, k:k_end],
                        B[k:k_end, j:j_end]
                    )
        
        return C
    
    @staticmethod
    def blocked_multiplication_manual(A, B, block_size=64):
        """
        Manual blocked multiplication without numpy.dot inside
        More control over cache behavior
        """
        n = A.shape[0]
        C = np.zeros((n, n))
        
        for ii in range(0, n, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, n, block_size):
                    # Process block
                    for i in range(ii, min(ii + block_size, n)):
                        for j in range(jj, min(jj + block_size, n)):
                            temp = 0.0
                            for k in range(kk, min(kk + block_size, n)):
                                temp += A[i, k] * B[k, j]
                            C[i, j] += temp
        
        return C
    
    @staticmethod
    def transpose_multiplication(A, B):
        """
        Optimized multiplication using B transpose
        Improves cache locality for B access
        """
        n = A.shape[0]
        B_T = B.T.copy()  # Transpose B for better cache access
        C = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                C[i, j] = np.dot(A[i, :], B_T[j, :])
        
        return C
    
    def benchmark_algorithm(self, algorithm_name: str, A, B, num_runs=3, warmup=1):
        """
        Benchmark a specific multiplication algorithm
        """
        algorithms = {
            'naive': self.naive_multiplication,
            'numpy': self.numpy_multiplication,
            'strassen': self.strassen_multiplication,
            'blocked': lambda a, b: self.blocked_multiplication(a, b, block_size=64),
            'blocked_manual': lambda a, b: self.blocked_multiplication_manual(a, b, block_size=64),
            'transpose': self.transpose_multiplication
        }
        
        if algorithm_name not in algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algo_func = algorithms[algorithm_name]
        n = A.shape[0]
        
        # Warmup
        for _ in range(warmup):
            _ = algo_func(A, B)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            C = algo_func(A, B)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        
        # Calculate GFlops: n^3 multiplications + n^3 additions = 2n^3 operations
        flops = 2 * n ** 3
        gflops = (flops / avg_time) / 1e9
        
        result = DenseResult(
            algorithm=algorithm_name,
            matrix_size=n,
            execution_time=avg_time,
            gflops=gflops
        )
        
        self.results.append(result)
        return result
    
    def compare_all_algorithms(self, sizes=[100, 200, 500], algorithms=None):
        """
        Compare all implemented algorithms
        """
        if algorithms is None:
            # Don't include naive for large sizes (too slow)
            algorithms = ['numpy', 'blocked', 'strassen', 'transpose']
        
        print("\n" + "="*70)
        print("DENSE MATRIX MULTIPLICATION COMPARISON")
        print("="*70)
        
        all_results = []
        
        for n in sizes:
            print(f"\nMatrix size: {n}x{n}")
            
            # Generate random matrices
            A = np.random.rand(n, n)
            B = np.random.rand(n, n)
            
            # Include naive only for small sizes
            test_algos = algorithms.copy()
            if n <= 200 and 'naive' not in test_algos:
                test_algos = ['naive'] + test_algos
            
            for algo in test_algos:
                try:
                    print(f"  Testing {algo}...", end=" ")
                    result = self.benchmark_algorithm(algo, A, B, num_runs=3)
                    all_results.append(result)
                    print(f"{result.execution_time:.6f}s ({result.gflops:.3f} GFlop/s)")
                except Exception as e:
                    print(f"Error: {e}")
        
        return all_results
    
    def compare_block_sizes(self, n=1000, block_sizes=[32, 64, 128, 256]):
        """
        Compare different block sizes for blocked multiplication
        """
        print("\n" + "="*70)
        print(f"BLOCK SIZE COMPARISON (n={n})")
        print("="*70)
        
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        results = []
        
        for bs in block_sizes:
            print(f"\nBlock size: {bs}")
            
            times = []
            for _ in range(3):
                start = time.perf_counter()
                C = self.blocked_multiplication(A, B, block_size=bs)
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            flops = 2 * n ** 3
            gflops = (flops / avg_time) / 1e9
            
            results.append({
                'block_size': bs,
                'time': avg_time,
                'gflops': gflops
            })
            
            print(f"  Time: {avg_time:.6f}s, GFlop/s: {gflops:.3f}")
        
        return results


def main():
    """Main function to run all dense optimization tests"""
    optimizer = OptimizedDenseMultiplication()
    
    print("="*70)
    print("OPTIMIZED DENSE MATRIX MULTIPLICATION")
    print("Task 2 - Big Data Course")
    print("="*70)
    
    # Test 1: Compare all algorithms
    print("\n### TEST 1: ALGORITHM COMPARISON ###")
    comparison_results = optimizer.compare_all_algorithms(
        sizes=[100, 200, 500, 1000],
        algorithms=['numpy', 'blocked', 'strassen', 'transpose']
    )
    
    # Test 2: Block size comparison
    print("\n### TEST 2: BLOCK SIZE OPTIMIZATION ###")
    block_results = optimizer.compare_block_sizes(
        n=1000,
        block_sizes=[32, 64, 128, 256]
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nTotal tests run: {len(optimizer.results)}")
    
    if optimizer.results:
        best = max(optimizer.results, key=lambda x: x.gflops)
        print(f"\nBest performance:")
        print(f"  Algorithm: {best.algorithm}")
        print(f"  Matrix size: {best.matrix_size}")
        print(f"  Performance: {best.gflops:.3f} GFlop/s")
        print(f"  Time: {best.execution_time:.6f}s")


if __name__ == "__main__":
    main()
