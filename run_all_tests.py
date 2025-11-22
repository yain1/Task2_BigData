"""
Main execution script for Task 2
Runs all benchmarks and generates visualizations
Task 2 - Big Data Course
Author: Yaín René Estrada Domínguez
"""

import sys
import json
import time
from datetime import datetime
import numpy as np

# Import our modules
from sparse_benchmark import SparseMatrixBenchmark
from optimized_dense import OptimizedDenseMultiplication
from visualization import BenchmarkVisualizer


def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def save_comparison_data(data, filename="comparison_data.json"):
    """Save dense vs sparse comparison data"""
    with open(f"results/{filename}", 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved comparison data to results/{filename}")


def main():
    """Main execution function"""
    
    start_time = time.time()
    
    print_header("TASK 2: OPTIMIZED MATRIX MULTIPLICATION AND SPARSE MATRICES")
    print(f"Execution started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Author: Yaín René Estrada Domínguez")
    print(f"Course: Big Data - Universidad de Las Palmas de Gran Canaria")
    
    # =========================================================================
    # PART 1: DENSE MATRIX OPTIMIZATIONS
    # =========================================================================
    
    print_header("PART 1: DENSE MATRIX MULTIPLICATION - OPTIMIZED ALGORITHMS")
    
    dense_optimizer = OptimizedDenseMultiplication()
    
    print("\n[1.1] Comparing optimization algorithms...")
    print("-" * 80)
    
    # Test different algorithms with increasing sizes
    dense_results = dense_optimizer.compare_all_algorithms(
        sizes=[100, 200, 500, 1000],
        algorithms=['numpy', 'blocked', 'strassen', 'transpose']
    )
    
    print("\n[1.2] Testing block size optimization...")
    print("-" * 80)
    
    block_results = dense_optimizer.compare_block_sizes(
        n=1000,
        block_sizes=[32, 64, 128, 256]
    )
    
    # Save dense results
    dense_results_dict = [r.to_dict() for r in dense_optimizer.results]
    with open("results/dense_results.json", 'w') as f:
        json.dump(dense_results_dict, f, indent=2)
    
    print("\n✓ Dense optimization tests completed")
    
    # =========================================================================
    # PART 2: SPARSE MATRIX OPERATIONS
    # =========================================================================
    
    print_header("PART 2: SPARSE MATRIX MULTIPLICATION")
    
    sparse_benchmark = SparseMatrixBenchmark()
    
    print("\n[2.1] Testing different sparsity levels...")
    print("-" * 80)
    
    # Compare performance across sparsity levels
    sparsity_comparison = sparse_benchmark.run_sparsity_comparison(
        sizes=[100, 500, 1000, 2000, 5000],
        sparsity_levels=[0.90, 0.95, 0.99, 0.999]
    )
    
    # Save comparison data
    save_comparison_data(sparsity_comparison, "sparsity_comparison.json")
    
    print("\n[2.2] Testing with mc2depi matrix (real-world epidemiology data)...")
    print("-" * 80)
    
    # Load and test mc2depi matrix
    try:
        mc2depi = sparse_benchmark.download_and_load_mc2depi()
        
        if mc2depi is not None:
            print("\nRunning algorithm comparison on mc2depi...")
            mc2depi_results = sparse_benchmark.run_algorithm_comparison(mc2depi)
            
            # Save mc2depi specific results
            mc2depi_dict = {
                'matrix_info': {
                    'name': 'mc2depi',
                    'shape': mc2depi.shape,
                    'nnz': int(mc2depi.nnz),
                    'sparsity': float(1 - mc2depi.nnz/(mc2depi.shape[0]*mc2depi.shape[1]))
                },
                'results': [r.to_dict() for r in mc2depi_results]
            }
            
            with open("results/mc2depi_results.json", 'w') as f:
                json.dump(mc2depi_dict, f, indent=2)
            
            print("\n✓ mc2depi tests completed")
        else:
            print("\n⚠ Could not load mc2depi matrix (network issue?)")
            print("  Continuing with other tests...")
            
    except Exception as e:
        print(f"\n⚠ Error with mc2depi matrix: {e}")
        print("  Continuing with other tests...")
    
    print("\n[2.3] Finding maximum matrix size...")
    print("-" * 80)
    
    # Test maximum size
    max_size_results = sparse_benchmark.test_maximum_size(
        sparsity=0.99,
        start_size=5000,
        max_time=30
    )
    
    # Save max size results
    with open("results/max_size_results.json", 'w') as f:
        json.dump(max_size_results, f, indent=2)
    
    print("\n✓ Maximum size tests completed")
    
    # Save all sparse results
    sparse_benchmark.save_results("sparse_results.json")
    
    # =========================================================================
    # PART 3: ANALYSIS AND VISUALIZATION
    # =========================================================================
    
    print_header("PART 3: GENERATING VISUALIZATIONS AND ANALYSIS")
    
    visualizer = BenchmarkVisualizer()
    
    print("\n[3.1] Loading results...")
    sparse_results = sparse_benchmark.results
    
    print("\n[3.2] Generating plots...")
    print("-" * 80)
    
    try:
        # Convert results to dict format for visualization
        sparse_results_dict = [r.to_dict() for r in sparse_results]
        
        visualizer.generate_all_plots(
            sparse_results=sparse_results_dict,
            dense_results=dense_results_dict,
            comparison_data=sparsity_comparison
        )
        
        print("\n✓ All visualizations generated successfully")
        
    except Exception as e:
        print(f"\n⚠ Error generating visualizations: {e}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print_header("EXECUTION SUMMARY")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📊 Results saved in:")
    print("   - results/sparse_results.json")
    print("   - results/dense_results.json")
    print("   - results/sparsity_comparison.json")
    print("   - results/mc2depi_results.json")
    print("   - results/max_size_results.json")
    
    print("\n📈 Figures saved in:")
    print("   - figures/sparsity_vs_performance.png")
    print("   - figures/memory_usage.png")
    print("   - figures/scalability.png")
    print("   - figures/algorithm_comparison.png")
    print("   - figures/dense_vs_sparse.png")
    print("   - figures/summary_table.txt")
    
    print("\n" + "="*80)
    print("  ✓ ALL TESTS COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    # Print key findings
    print_header("KEY FINDINGS")
    
    if sparse_results:
        # Find best sparse performance
        best_sparse = max(sparse_results, key=lambda x: x.gflops)
        print(f"🏆 Best Sparse Performance:")
        print(f"   - Algorithm: {best_sparse.algorithm}")
        print(f"   - Matrix size: {best_sparse.matrix_size}x{best_sparse.matrix_size}")
        print(f"   - Sparsity: {best_sparse.sparsity*100:.2f}%")
        print(f"   - Performance: {best_sparse.gflops:.4f} GFlop/s")
        print(f"   - Time: {best_sparse.execution_time:.6f} seconds")
    
    if dense_optimizer.results:
        # Find best dense performance
        best_dense = max(dense_optimizer.results, key=lambda x: x.gflops)
        print(f"\n🏆 Best Dense Performance:")
        print(f"   - Algorithm: {best_dense.algorithm}")
        print(f"   - Matrix size: {best_dense.matrix_size}x{best_dense.matrix_size}")
        print(f"   - Performance: {best_dense.gflops:.4f} GFlop/s")
        print(f"   - Time: {best_dense.execution_time:.6f} seconds")
    
    # Sparsity insights
    if sparsity_comparison:
        high_sparsity = [r for r in sparsity_comparison if r['sparsity'] >= 0.99]
        if high_sparsity:
            avg_speedup = np.mean([r['speedup'] for r in high_sparsity if r['speedup']])
            print(f"\n💡 Sparsity Insights:")
            print(f"   - Average speedup at 99%+ sparsity: {avg_speedup:.2f}x")
            print(f"   - Memory savings at high sparsity: ~{1/(1-0.99)*100:.0f}% reduction")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
