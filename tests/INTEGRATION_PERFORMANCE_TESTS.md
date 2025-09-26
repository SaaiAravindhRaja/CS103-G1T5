# Integration and Performance Tests

This document describes the comprehensive integration and performance test suite for the SVD Image Compression system.

## Overview

The test suite covers four main areas:
1. **CLI Workflow Integration Tests** - End-to-end command-line interface testing
2. **Batch Processing Integration Tests** - Systematic batch experiment testing
3. **Performance Benchmarks** - Compression algorithm performance analysis
4. **Memory Profiling Tests** - Memory usage and leak detection

## Test Categories

### 1. CLI Workflow Integration Tests (`test_cli_workflows.py`)

Tests the complete command-line interface functionality:

- **Basic CLI Functionality**: Tests core CLI demo operations
- **Argument Parsing**: Validates command-line argument handling
- **Error Handling**: Tests graceful error handling for invalid inputs
- **Output Formats**: Verifies CSV and image output generation
- **Batch Processing Workflow**: Tests complete batch processing pipeline
- **Resume Functionality**: Tests checkpoint and resume capabilities
- **Parallel Processing**: Validates parallel execution modes
- **Configuration Validation**: Tests parameter validation
- **Memory Management**: Monitors memory usage during CLI operations
- **Stress Testing**: Tests with large datasets and many parameters
- **Edge Cases**: Tests boundary conditions and small images
- **Robustness**: Tests handling of corrupted or invalid data
- **Performance Monitoring**: Validates timing accuracy and reporting

### 2. Batch Processing Integration Tests (`test_batch_processing.py`)

Tests the systematic batch experiment framework:

- **Complete Batch Workflow**: End-to-end batch processing pipeline
- **Experiment Resumption**: Checkpoint saving and loading functionality
- **Result Manager**: Result storage, export, and analysis features
- **Parallel Processing**: Multi-threaded batch execution
- **Error Handling**: Graceful handling of compression failures
- **Checkpoint Functionality**: Incremental progress saving
- **Experiment Summary**: Statistical analysis and reporting
- **Result Export**: Multiple format export capabilities
- **Large Scale Processing**: High-volume experiment handling
- **Data Integrity**: Validation of result accuracy and completeness
- **Scalability**: Performance across different workload sizes
- **Resource Cleanup**: Memory and handle management
- **Fault Tolerance**: Recovery from errors and edge cases

### 3. Performance Benchmarks (`test_compression_benchmarks.py`)

Comprehensive performance analysis of compression algorithms:

- **Timing Scalability**: Performance across different image sizes
- **Memory Usage Scalability**: Memory consumption patterns
- **K-Value Performance Impact**: Performance variation with compression levels
- **Batch Compression Performance**: Throughput analysis
- **Memory Leak Detection**: Long-running operation stability
- **Concurrent Compression**: Multi-threaded performance
- **Large Image Performance**: Handling of high-resolution images
- **Quality vs Performance Tradeoff**: Analysis of compression efficiency
- **Numerical Stability**: Performance with challenging numerical cases
- **Algorithm Efficiency**: Comparative analysis across scenarios
- **Throughput Analysis**: Pixels-per-second measurements
- **Scalability Limits**: Behavior at maximum k-values
- **Memory Efficiency Benchmarks**: Memory usage optimization
- **Performance Regression**: Baseline performance validation

### 4. Memory Profiling Tests (`test_memory_profiling.py`)

Detailed memory usage analysis and leak detection:

- **Compression Memory Usage**: Memory consumption during operations
- **Memory Leak Detection**: Long-term stability testing
- **Peak Memory Usage**: Maximum memory consumption analysis
- **Concurrent Memory Usage**: Multi-threaded memory behavior
- **Memory Efficiency by K-Value**: Memory usage across compression levels
- **Component Memory Usage**: Individual module memory analysis
- **Memory Cleanup After Errors**: Error handling memory management
- **Large Batch Memory Efficiency**: Batch processing memory patterns
- **Memory Fragmentation Analysis**: Memory allocation patterns
- **Memory Pressure Handling**: Behavior under memory constraints
- **Memory Usage Patterns**: Analysis across different scenarios
- **Memory Profiling Accuracy**: Validation of measurement tools
- **Memory Leak Stress Testing**: Intensive operation stability
- **Memory Optimization Verification**: Validation of optimizations

## Running the Tests

### Individual Test Suites

```bash
# CLI workflow tests
python -m pytest tests/integration/test_cli_workflows.py -v

# Batch processing tests
python -m pytest tests/integration/test_batch_processing.py -v

# Performance benchmarks
python -m pytest tests/performance/test_compression_benchmarks.py -v

# Memory profiling tests
python -m pytest tests/performance/test_memory_profiling.py -v
```

### Complete Test Suite

```bash
# Run all integration and performance tests
python tests/run_integration_performance_tests.py

# Or using pytest directly
python -m pytest tests/integration/ tests/performance/ -v
```

### Specific Test Categories

```bash
# Run only integration tests
python -m pytest tests/integration/ -v

# Run only performance tests
python -m pytest tests/performance/ -v

# Run with coverage
python -m pytest tests/integration/ tests/performance/ --cov=src --cov-report=html
```

## Test Requirements

### System Requirements
- Python 3.8+
- At least 4GB RAM (for memory profiling tests)
- Multi-core CPU (for parallel processing tests)

### Dependencies
- pytest
- numpy
- scipy
- scikit-image
- pillow
- pandas
- psutil (for memory monitoring)
- tqdm (for progress bars)

### Test Data
Tests automatically generate synthetic test data including:
- Various image sizes (64x64 to 512x512)
- Different image types (grayscale, RGB)
- Multiple datasets (portraits, landscapes, textures)
- Edge cases (very small images, constant images, etc.)

## Performance Baselines

### Expected Performance Ranges
- **64x64 image compression**: < 1 second
- **256x256 image compression**: < 10 seconds
- **Memory usage**: < 200MB for typical operations
- **Batch processing**: > 0.1 megapixels/second throughput

### Memory Usage Guidelines
- **Peak memory**: < 10x image size for compression
- **Memory cleanup**: > 90% memory recovery after operations
- **Memory leaks**: < 50MB increase over 100 operations

## Troubleshooting

### Common Issues

1. **Memory Tests Failing**: Adjust memory thresholds based on system capabilities
2. **Timeout Errors**: Increase timeout values for slower systems
3. **Parallel Processing Issues**: Reduce worker count for systems with limited cores
4. **Platform-Specific Failures**: Some memory measurements may vary by OS

### Test Configuration

Tests can be configured by modifying constants at the top of test files:
- Memory thresholds
- Timeout values
- Image sizes
- Number of test iterations

## Continuous Integration

These tests are designed to run in CI environments with:
- Reasonable execution times (< 30 minutes total)
- Robust error handling
- Platform-independent assertions
- Configurable resource limits

## Test Coverage

The integration and performance tests provide coverage for:
- All major system components
- Error handling paths
- Performance characteristics
- Memory management
- Scalability limits
- Real-world usage scenarios

This comprehensive test suite ensures the SVD Image Compression system is robust, performant, and ready for production use.