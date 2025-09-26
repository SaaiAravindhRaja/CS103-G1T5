#!/usr/bin/env python3
"""
Comprehensive test runner for integration and performance tests.

This script runs all integration and performance tests with proper
reporting and validation of the SVD image compression system.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_suite(test_path: str, description: str) -> bool:
    """Run a test suite and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pytest', 
            test_path, 
            '-v', 
            '--tb=short',
            '--durations=10'
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Duration: {duration:.2f} seconds")
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            print(f"Tests completed successfully in {duration:.2f}s")
            return True
        else:
            print(f"❌ {description} - FAILED")
            print("STDOUT:")
            print(result.stdout)
            print("STDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        print("Tests exceeded 5 minute timeout")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR")
        print(f"Error running tests: {e}")
        return False


def main():
    """Run comprehensive integration and performance test suite."""
    print("SVD Image Compression - Integration & Performance Test Suite")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        ("tests/integration/test_cli_workflows.py", "CLI Workflow Integration Tests"),
        ("tests/integration/test_batch_processing.py", "Batch Processing Integration Tests"),
        ("tests/integration/test_dataset_manager.py", "Dataset Manager Integration Tests"),
        ("tests/integration/test_end_to_end_workflows.py", "End-to-End Workflow Tests"),
        ("tests/performance/test_compression_benchmarks.py", "Compression Performance Benchmarks"),
        ("tests/performance/test_memory_profiling.py", "Memory Profiling Tests"),
    ]
    
    results = []
    total_start_time = time.time()
    
    # Run each test suite
    for test_path, description in test_suites:
        success = run_test_suite(test_path, description)
        results.append((description, success))
    
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUITE SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {description}")
    
    print(f"\nOverall Results: {passed}/{total} test suites passed")
    print(f"Total Duration: {total_duration:.2f} seconds")
    
    if passed == total:
        print("\n🎉 All integration and performance tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())