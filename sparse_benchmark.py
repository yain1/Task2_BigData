"""
Sparse Matrix Benchmark Suite
Task 2 - Big Data Course
Author: Yaín René Estrada Domínguez
"""

import numpy as np
import time
from scipy.sparse import csr_matrix, random as sparse_random
from scipy.io import mmread
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results"""
    matrix_size: int
    sparsity: float
    nnz: int
    execution_time: float
    std_time: float
    gflops: float
    memory_mb: float
    format_type: str
    algorithm: str
    
    def to_dict(self):
        return asdict(self)


class SparseMatrixBenchmark:
    """Comprehensive benchmark suite for sparse matrix operations"""
    
    def __init__(self, results_dir="results"):
        self.results = []
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def load_mc2depi(self):
        """Load the mc2depi matrix"""
        filepath = "mc2depi.mtx"
        
        if not os.path.exists(filepath):
            print(f"❌ File {filepath} not found!")
            return None
        
        print(f"Loading matrix from {filepath}...")
        try:
            A = mmread(filepath)
            
            if not isinstance(A, csr_matrix):
                A = csr_matrix(A)
            
            print(f"\nMatrix loaded successfully:")
            print(f"  Shape: {A.shape}")
            print(f"  Non-zeros: {A.nnz:,}")
            print(f"  Sparsity: {(1 - A.nnz/(A.shape[0]*A.shape[1]))*100:.4f}%")
            print(f"  Avg nnz/row: {A.nnz/A.shape[0]:.2f}")
            
            return A
        except Exception as e:
            print(f"❌ Error loading matrix: {e}")
            return None
    
    def create_sparse_matrix(self, n: int, sparsity: float = 0.99, seed: int = 42):
        """
        Create a random sparse matrix with specified sparsity
        """
        np.random.seed(seed)
        density = 1 - sparsity
        A = sparse_random(n, n, density=density, format='csr', dtype=np.float64, random_state=seed)
        return A
    
    def csr_spmv_basic(self, A_csr, x):
        """Basic CSR Sparse Matrix-Vector Multiplication"""
        return A_csr.dot(x)
    
    def csr_spmv_manual(self, data, indices, indptr, x):
        """Manual implementation of CSR SpMV"""
        m = len(indptr) - 1
        y = np.zeros(m, dtype=np.float64)
        
        for i in range(m):
            y0 = 0.0
            for k in range(indptr[i], indptr[i+1]):
                y0 += data[k] * x[indices[k]]
            y[i] = y0
        
        return y
    
    def benchmark_spmv(self, A, algorithm="scipy", num_runs=5, warmup=1):
        """Benchmark sparse matrix-vector multiplication"""
        n = A.shape[1]
        x = np.random.rand(n)
        
        # Warmup runs
        for _ in range(warmup):
            if algorithm == "manual":
                _ = self.csr_spmv_manual(A.data, A.indices, A.indptr, x)
            else:
                _ = A.dot(x)
        
        # Benchmark runs
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            
            if algorithm == "manual":
                y = self.csr_spmv_manual(A.data, A.indices, A.indptr, x)
            elif algorithm == "scipy":
                y = self.csr_spmv_basic(A, x)
            else:  # optimized
                y = A @ x
            
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate performance metrics
        flops = 2 * A.nnz
        gflops = (flops / avg_time) / 1e9
        
        # Calculate memory usage
        memory_mb = (A.data.nbytes + A.indices.nbytes + A.indptr.nbytes) / (1024**2)
        
        sparsity = 1 - (A.nnz / (A.shape[0] * A.shape[1]))
        
        result = BenchmarkResult(
            matrix_size=A.shape[0],
            sparsity=sparsity,
            nnz=A.nnz,
            execution_time=avg_time,
            std_time=std_time,
            gflops=gflops,
            memory_mb=memory_mb,
            format_type="CSR",
            algorithm=algorithm
        )
        
        self.results.append(result)
        return result
    
    def benchmark_dense_vs_sparse(self, n: int, sparsity: float, num_runs=3):
        """Compare dense vs sparse matrix-vector multiplication"""
        print(f"\n  Testing n={n}, sparsity={sparsity*100:.1f}%...")
        
        # Create sparse matrix
        A_sparse = self.create_sparse_matrix(n, sparsity)
        x = np.random.rand(n)
        
        # Benchmark sparse
        sparse_result = self.benchmark_spmv(A_sparse, algorithm="scipy", num_runs=num_runs)
        
        # Benchmark dense (only for smaller matrices)
        dense_time = None
        dense_gflops = None
        speedup = None
        memory_ratio = None
        
        if n <= 2000:  # Only test dense for reasonable sizes
            A_dense = A_sparse.toarray()
            
            # Warmup
            _ = np.dot(A_dense, x)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                y = np.dot(A_dense, x)
                end = time.perf_counter()
                times.append(end - start)
            
            dense_time = np.mean(times)
            dense_flops = 2 * n * n
            dense_gflops = (dense_flops / dense_time) / 1e9
            speedup = dense_time / sparse_result.execution_time
            
            dense_memory_mb = (n * n * 8) / (1024**2)
            memory_ratio = dense_memory_mb / sparse_result.memory_mb
        
        return {
            'n': n,
            'sparsity': sparsity,
            'nnz': A_sparse.nnz,
            'sparse_time': sparse_result.execution_time,
            'sparse_gflops': sparse_result.gflops,
            'sparse_memory_mb': sparse_result.memory_mb,
            'dense_time': dense_time,
            'dense_gflops': dense_gflops,
            'speedup': speedup,
            'memory_ratio': memory_ratio
        }
    
    def run_sparsity_comparison(self, sizes=[100, 500, 1000, 2000, 5000], 
                               sparsity_levels=[0.90, 0.95, 0.99, 0.999]):
        """Compare performance across different sparsity levels"""
        print("\n" + "="*70)
        print("SPARSITY LEVEL COMPARISON")
        print("="*70)
        
        all_results = []
        
        for n in sizes:
            print(f"\nMatrix size: {n}x{n}")
            for sparsity in sparsity_levels:
                result = self.benchmark_dense_vs_sparse(n, sparsity)
                all_results.append(result)
                
                print(f"    Sparsity {sparsity*100:.1f}%: "
                      f"Sparse={result['sparse_time']:.6f}s ({result['sparse_gflops']:.3f} GFlop/s), "
                      f"Memory={result['sparse_memory_mb']:.2f} MB", end="")
                
                if result['speedup']:
                    print(f", Speedup={result['speedup']:.2f}x")
                else:
                    print()
        
        return all_results
    
    def run_algorithm_comparison(self, A):
        """Compare different SpMV implementations"""
        print("\n" + "="*70)
        print("ALGORITHM COMPARISON")
        print("="*70)
        
        algorithms = ["scipy", "manual", "optimized"]
        results = []
        
        for algo in algorithms:
            print(f"\nTesting algorithm: {algo}")
            result = self.benchmark_spmv(A, algorithm=algo, num_runs=5)
            results.append(result)
            
            print(f"  Time: {result.execution_time:.6f}s ± {result.std_time:.6f}s")
            print(f"  Performance: {result.gflops:.4f} GFlop/s")
        
        return results
    
    def test_maximum_size(self, sparsity=0.99, start_size=1000, max_time=60):
        """Find the maximum matrix size that can be handled efficiently"""
        print("\n" + "="*70)
        print("MAXIMUM SIZE TEST")
        print("="*70)
        
        sizes_tested = []
        n = start_size
        
        while True:
            print(f"\nTesting size {n}x{n}...")
            
            try:
                A = self.create_sparse_matrix(n, sparsity)
                result = self.benchmark_spmv(A, num_runs=3)
                
                sizes_tested.append({
                    'size': n,
                    'time': result.execution_time,
                    'gflops': result.gflops,
                    'memory_mb': result.memory_mb
                })
                
                print(f"  Success: {result.execution_time:.6f}s, {result.gflops:.4f} GFlop/s")
                
                if result.execution_time > max_time:
                    print(f"\n  Reached time limit ({max_time}s)")
                    break
                
                if n > 20000:
                    break
                n = int(n * 1.5)

                
            except MemoryError:
                print(f"  Memory error at size {n}")
                break
            except Exception as e:
                print(f"  Error: {e}")
                break
        
        return sizes_tested
    
    def save_results(self, filename="benchmark_results.json"):
        """Save benchmark results to JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        
        results_dict = [r.to_dict() for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to {filepath}")
    
    def print_summary(self):
        """Print summary of all benchmark results"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        
        if not self.results:
            print("No results to display")
            return
        
        print(f"\nTotal benchmarks run: {len(self.results)}")
        print(f"\nPerformance statistics:")
        
        times = [r.execution_time for r in self.results]
        gflops = [r.gflops for r in self.results]
        
        print(f"  Execution time: min={min(times):.6f}s, max={max(times):.6f}s, avg={np.mean(times):.6f}s")
        print(f"  Performance: min={min(gflops):.4f}, max={max(gflops):.4f}, avg={np.mean(gflops):.4f} GFlop/s")


if __name__ == "__main__":
    benchmark = SparseMatrixBenchmark()
    
    print("="*70)
    print("SPARSE MATRIX BENCHMARK SUITE")
    print("Task 2 - Big Data Course")
    print("="*70)
    
    # Test 1: Sparsity comparison
    print("\n\n### TEST 1: SPARSITY LEVEL COMPARISON ###")
    sparsity_results = benchmark.run_sparsity_comparison(
        sizes=[100, 500, 1000, 2000],
        sparsity_levels=[0.90, 0.95, 0.99, 0.999]
    )
    
    # Test 2: Load and test mc2depi matrix
    print("\n\n### TEST 2: MC2DEPI MATRIX (REAL-WORLD DATA) ###")
    mc2depi = benchmark.load_mc2depi()
    
    if mc2depi is not None:
        mc2depi_results = benchmark.run_algorithm_comparison(mc2depi)
    
    # Test 3: Maximum size test
    print("\n\n### TEST 3: MAXIMUM MATRIX SIZE TEST ###")
    max_size_results = benchmark.test_maximum_size(sparsity=0.99, start_size=5000, max_time=30)
    
    # Print summary and save results
    benchmark.print_summary()
    benchmark.save_results()
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
