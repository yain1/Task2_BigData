"""
Visualization and Analysis Tools
Task 2 - Big Data Course
Author: Yaín René Estrada Domínguez
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List, Dict


class BenchmarkVisualizer:
    """Create visualizations for benchmark results"""
    
    def __init__(self, results_dir="results", output_dir="figures"):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for better-looking plots
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def load_results(self, filename="benchmark_results.json"):
        """Load results from JSON file"""
        filepath = os.path.join(self.results_dir, filename)
        
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        return results
    
    def plot_sparsity_vs_performance(self, results, save=True):
        """
        Plot: Performance vs Sparsity Level
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by matrix size
        sizes = sorted(set(r['matrix_size'] for r in results))
        
        for size in sizes:
            size_results = [r for r in results if r['matrix_size'] == size]
            sparsities = [r['sparsity'] * 100 for r in size_results]
            gflops = [r['gflops'] for r in size_results]
            times = [r['execution_time'] for r in size_results]
            
            ax1.plot(sparsities, gflops, marker='o', label=f'n={size}', linewidth=2)
            ax2.plot(sparsities, times, marker='s', label=f'n={size}', linewidth=2)
        
        ax1.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax1.set_ylabel('Performance (GFlop/s)', fontsize=12)
        ax1.set_title('Performance vs Sparsity Level', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax2.set_ylabel('Execution Time (s)', fontsize=12)
        ax2.set_title('Execution Time vs Sparsity Level', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'sparsity_vs_performance.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_memory_usage(self, results, save=True):
        """
        Plot: Memory usage comparison
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sizes = sorted(set(r['matrix_size'] for r in results))
        sparsity_levels = sorted(set(r['sparsity'] for r in results))
        
        x = np.arange(len(sizes))
        width = 0.2
        
        for i, sparsity in enumerate(sparsity_levels):
            memory = []
            for size in sizes:
                matching = [r for r in results 
                           if r['matrix_size'] == size and abs(r['sparsity'] - sparsity) < 0.001]
                if matching:
                    memory.append(matching[0]['memory_mb'])
                else:
                    memory.append(0)
            
            ax.bar(x + i*width, memory, width, 
                   label=f'Sparsity {sparsity*100:.1f}%', alpha=0.8)
        
        ax.set_xlabel('Matrix Size', fontsize=12)
        ax.set_ylabel('Memory Usage (MB)', fontsize=12)
        ax.set_title('Memory Usage vs Matrix Size and Sparsity', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(sparsity_levels)-1) / 2)
        ax.set_xticklabels([f'{s}x{s}' for s in sizes])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'memory_usage.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_scalability(self, results, save=True):
        """
        Plot: Scalability analysis (size vs time)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by sparsity
        sparsities = sorted(set(r['sparsity'] for r in results))
        
        for sparsity in sparsities:
            sparsity_results = [r for r in results if abs(r['sparsity'] - sparsity) < 0.001]
            sparsity_results.sort(key=lambda x: x['matrix_size'])
            
            sizes = [r['matrix_size'] for r in sparsity_results]
            times = [r['execution_time'] for r in sparsity_results]
            gflops = [r['gflops'] for r in sparsity_results]
            
            ax1.plot(sizes, times, marker='o', label=f'{sparsity*100:.1f}%', linewidth=2)
            ax2.plot(sizes, gflops, marker='s', label=f'{sparsity*100:.1f}%', linewidth=2)
        
        ax1.set_xlabel('Matrix Size', fontsize=12)
        ax1.set_ylabel('Execution Time (s)', fontsize=12)
        ax1.set_title('Scalability: Size vs Time', fontsize=14, fontweight='bold')
        ax1.legend(title='Sparsity')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        
        ax2.set_xlabel('Matrix Size', fontsize=12)
        ax2.set_ylabel('Performance (GFlop/s)', fontsize=12)
        ax2.set_title('Scalability: Size vs Performance', fontsize=14, fontweight='bold')
        ax2.legend(title='Sparsity')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'scalability.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_algorithm_comparison(self, results, save=True):
        """
        Plot: Algorithm comparison
        """
        if not results:
            print("No results to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithms = sorted(set(r['algorithm'] for r in results))
        sizes = sorted(set(r['matrix_size'] for r in results))
        
        x = np.arange(len(algorithms))
        width = 0.15
        
        for i, size in enumerate(sizes):
            gflops = []
            for algo in algorithms:
                matching = [r for r in results 
                           if r['algorithm'] == algo and r['matrix_size'] == size]
                if matching:
                    gflops.append(matching[0]['gflops'])
                else:
                    gflops.append(0)
            
            ax.bar(x + i*width, gflops, width, label=f'n={size}', alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=12)
        ax.set_ylabel('Performance (GFlop/s)', fontsize=12)
        ax.set_title('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(sizes)-1) / 2)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'algorithm_comparison.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def plot_dense_vs_sparse_speedup(self, comparison_data, save=True):
        """
        Plot: Dense vs Sparse speedup
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Group by size
        sizes = sorted(set(r['n'] for r in comparison_data))
        
        for size in sizes:
            size_data = [r for r in comparison_data if r['n'] == size and r['speedup']]
            if not size_data:
                continue
            
            sparsities = [r['sparsity'] * 100 for r in size_data]
            speedups = [r['speedup'] for r in size_data]
            memory_ratios = [r['memory_ratio'] for r in size_data]
            
            ax1.plot(sparsities, speedups, marker='o', label=f'n={size}', linewidth=2)
            ax2.plot(sparsities, memory_ratios, marker='s', label=f'n={size}', linewidth=2)
        
        ax1.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax1.set_ylabel('Speedup (Dense/Sparse)', fontsize=12)
        ax1.set_title('Sparse vs Dense: Speedup', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Break-even')
        
        ax2.set_xlabel('Sparsity Level (%)', fontsize=12)
        ax2.set_ylabel('Memory Ratio (Dense/Sparse)', fontsize=12)
        ax2.set_title('Memory Savings', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'dense_vs_sparse.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
    
    def create_summary_table(self, results, save=True):
        """
        Create a summary table of results
        """
        print("\n" + "="*90)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*90)
        print(f"{'Size':<10} {'Sparsity':<12} {'NNZ':<12} {'Time (s)':<12} "
              f"{'GFlop/s':<12} {'Memory (MB)':<12}")
        print("-"*90)
        
        for r in sorted(results, key=lambda x: (x['matrix_size'], x['sparsity'])):
            print(f"{r['matrix_size']:<10} {r['sparsity']*100:>10.2f}% "
                  f"{r['nnz']:<12} {r['execution_time']:>10.6f}  "
                  f"{r['gflops']:>10.4f}  {r['memory_mb']:>10.2f}")
        
        if save:
            filepath = os.path.join(self.output_dir, 'summary_table.txt')
            with open(filepath, 'w') as f:
                f.write("BENCHMARK RESULTS SUMMARY\n")
                f.write("="*90 + "\n")
                f.write(f"{'Size':<10} {'Sparsity':<12} {'NNZ':<12} {'Time (s)':<12} "
                       f"{'GFlop/s':<12} {'Memory (MB)':<12}\n")
                f.write("-"*90 + "\n")
                
                for r in sorted(results, key=lambda x: (x['matrix_size'], x['sparsity'])):
                    f.write(f"{r['matrix_size']:<10} {r['sparsity']*100:>10.2f}% "
                           f"{r['nnz']:<12} {r['execution_time']:>10.6f}  "
                           f"{r['gflops']:>10.4f}  {r['memory_mb']:>10.2f}\n")
            
            print(f"\nSaved: {filepath}")
    
    def generate_all_plots(self, sparse_results=None, dense_results=None, comparison_data=None):
        """
        Generate all visualization plots
        """
        print("\n" + "="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        
        if sparse_results:
            print("\n1. Sparsity vs Performance plots...")
            self.plot_sparsity_vs_performance(sparse_results)
            
            print("\n2. Memory usage plots...")
            self.plot_memory_usage(sparse_results)
            
            print("\n3. Scalability plots...")
            self.plot_scalability(sparse_results)
            
            print("\n4. Summary table...")
            self.create_summary_table(sparse_results)
        
        if dense_results:
            print("\n5. Algorithm comparison plots...")
            self.plot_algorithm_comparison(dense_results)
        
        if comparison_data:
            print("\n6. Dense vs Sparse comparison...")
            self.plot_dense_vs_sparse_speedup(comparison_data)
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS GENERATED")
        print("="*70)


if __name__ == "__main__":
    visualizer = BenchmarkVisualizer()
    
    # Try to load and visualize results
    try:
        results = visualizer.load_results()
        visualizer.generate_all_plots(sparse_results=results)
    except FileNotFoundError:
        print("No results file found. Please run the benchmark first.")
