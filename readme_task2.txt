# Task 2: Optimized Matrix Multiplication and Sparse Matrices

**Course:** Big Data  
**Author:** Yaín René Estrada Domínguez  
**University:** Universidad de Las Palmas de Gran Canaria  
**Date:** November 2024

## 📋 Overview

This project implements and benchmarks optimized matrix multiplication algorithms, with special focus on sparse matrix operations. We compare multiple dense optimization techniques and demonstrate the dramatic performance benefits of sparse matrix formats for high-sparsity data.

## 🎯 Objectives

1. Implement optimized dense matrix multiplication algorithms
2. Implement sparse matrix operations using CSR format
3. Analyze performance across different sparsity levels (90%, 95%, 99%, 99.9%)
4. Test on the real-world `mc2depi` matrix from SuiteSparse Collection
5. Compare memory usage and computational performance
6. Determine maximum matrix sizes that can be handled efficiently

## 📁 Project Structure

```
Task2_BigData/
├── descarga.py                # Download mc2depi matrix
├── sparse_benchmark.py        # Sparse matrix benchmark suite
├── optimized_dense.py         # Dense optimization algorithms
├── visualization.py           # Results visualization
├── run_all_tests.py          # Main execution script
├── results/                  # Benchmark results (JSON)
│   ├── sparse_results.json
│   ├── dense_results.json
│   ├── mc2depi_results.json
│   └── sparsity_comparison.json
├── figures/                  # Generated plots
│   ├── sparsity_vs_performance.png
│   ├── memory_usage.png
│   ├── scalability.png
│   ├── algorithm_comparison.png
│   └── dense_vs_sparse.png
├── Task2_Report.pdf          # Full report
└── README.md
```

## 🛠️ Requirements

```bash
Python 3.8+
numpy >= 1.20.0
scipy >= 1.7.0
matplotlib >= 3.4.0
requests >= 2.26.0
```

Install dependencies:
```bash
pip install numpy scipy matplotlib requests
```

## 🚀 Quick Start

### Run All Benchmarks

```bash
python run_all_tests.py
```

This will:
1. Run dense matrix optimization comparisons
2. Test sparse matrices at different sparsity levels
3. Download and test the mc2depi matrix
4. Generate all visualizations
5. Save results to `results/` directory

### Individual Components

**Dense Matrix Benchmarks:**
```bash
python optimized_dense.py
```

**Sparse Matrix Benchmarks:**
```bash
python sparse_benchmark.py
```

**Generate Visualizations:**
```bash
python visualization.py
```

## 📊 Implemented Algorithms

### Dense Matrix Multiplication

1. **NumPy/BLAS** - Highly optimized baseline
2. **Strassen's Algorithm** - O(n^2.807) divide-and-conquer
3. **Blocked Multiplication** - Cache-optimized tiling
4. **Transpose Optimization** - Improved memory access patterns

### Sparse Matrix Operations

1. **CSR Format** - Compressed Sparse Row
2. **Sparse Matrix-Vector Multiplication (SpMV)**
3. **Manual vs Optimized implementations**

## 📈 Key Results

### Dense Optimization Performance (n=1000)

| Algorithm | Time (s) | GFlop/s |
|-----------|----------|---------|
| NumPy (BLAS) | 0.019 | 52.18 |
| Blocked | 0.020 | 48.92 |
| Strassen | 0.020 | 50.65 |
| Transpose | 0.022 | 46.21 |

### Sparse Matrix Performance (n=5000)

| Sparsity | Time (s) | Speedup | Memory Savings |
|----------|----------|---------|----------------|
| 90% | 0.052 | 4.1× | 10× |
| 95% | 0.026 | 8.2× | 20× |
| 99% | 0.005 | 41× | 100× |
| 99.9% | 0.0005 | 408× | 1000× |

### mc2depi Matrix (Real-World Test)

- **Dimensions:** 526,185 × 526,185
- **Non-zeros:** 2,100,225
- **Sparsity:** 99.6%
- **Execution time:** 3.4 ms
- **Memory usage:** 24.2 MB (vs 2,214 GB for dense!)
- **Performance:** 1.24 GFlop/s

## 🔍 Analysis Highlights

### Why Sparse Matrices Matter

1. **Dramatic Memory Savings:** 50-1000× reduction for high sparsity
2. **Computational Efficiency:** Only process non-zero elements
3. **Enables Large-Scale Problems:** Handle matrices too large for dense format
4. **Real-World Relevance:** Many applications naturally produce sparse data

### Optimal Use Cases for Sparse Formats

- Sparsity > 95%: Significant benefits
- Sparsity > 99%: Dramatic improvements (40-400× speedup)
- Applications: graphs, NLP, FEM, social networks, epidemiology

## 📊 Visualizations

All generated plots are saved in `figures/`:

1. **sparsity_vs_performance.png** - Performance across sparsity levels
2. **memory_usage.png** - Memory consumption comparison
3. **scalability.png** - Matrix size scaling analysis
4. **algorithm_comparison.png** - Dense algorithm comparison
5. **dense_vs_sparse.png** - Speedup analysis

## 🧪 Reproducibility

### System Configuration

- **Processor:** Intel Core i7-11700H
- **Memory:** 16 GB RAM
- **OS:** Windows 11
- **Python:** 3.11.x

### Running Tests

All tests use fixed random seeds for reproducibility:
```python
np.random.seed(42)
```

Each benchmark runs multiple trials with warmup:
- Warmup runs: 1
- Benchmark runs: 3-5
- Results: averaged with standard deviation

## 📚 References

1. Williams, S., et al. (2007). "Optimization of sparse matrix-vector multiplication on emerging multicore platforms"
2. Strassen, V. (1969). "Gaussian elimination is not optimal"
3. Davis, T. A., & Hu, Y. (2011). "The University of Florida sparse matrix collection"
4. SciPy Documentation: https://scipy.org/

## 📖 Report

Full detailed report available in `Task2_Report.pdf` including:
- Comprehensive methodology
- Detailed results and analysis
- Performance comparisons
- Conclusions and future work

## 🔗 Related Work

- **Task 1:** Basic matrix multiplication in C, Java, and Python
- **Paper Reference:** Williams et al. SpMV optimization paper (included)
- **Matrix Source:** [SuiteSparse Matrix Collection](https://sparse.tamu.edu/)

## 📝 Notes

### Memory Considerations

For very large sparse matrices:
- CSR format: ~12 bytes per non-zero element
- Dense format: 8 bytes per element (all n²)
- Break-even point: ~67% sparsity for CSR vs dense

### Performance Tips

1. Use SciPy's optimized implementations when possible
2. Choose format based on operation: CSR for row ops, CSC for column ops
3. Consider sparsity pattern and operation frequency
4. Profile before optimizing - memory often matters more than speed

## 🐛 Known Issues

- Very large matrices (>100K × 100K with low sparsity) may exceed memory
- Windows users: ensure sufficient virtual memory for large matrices
- macOS: matplotlib may require `TkAgg` backend

## 🤝 Contributing

This is an individual assignment, but suggestions welcome via issues.

## 📧 Contact

**Author:** Yaín René Estrada Domínguez  
**University:** Universidad de Las Palmas de Gran Canaria  
**Course:** Big Data - Grado en Ciencia e Ingeniería de Datos

## 📄 License

Academic use only - Universidad de Las Palmas de Gran Canaria

---

**Last Updated:** November 2024  
**Status:** ✅ Completed
