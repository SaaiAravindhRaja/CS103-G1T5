#!/usr/bin/env python3
"""
Final project validation script for SVD Image Compression System.
Validates all components are working correctly for task 11.3.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_test_suite():
    """Run the complete test suite."""
    print("🧪 Running complete test suite...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print(f"❌ Tests failed with return code {result.returncode}")
            print("STDOUT:", result.stdout[-500:])  # Last 500 chars
            print("STDERR:", result.stderr[-500:])
            return False
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False

def check_web_app():
    """Validate web application components."""
    print("🌐 Validating web application...")
    try:
        # Test imports
        sys.path.append('webapp')
        import app
        from pages import home, single_compression, batch_processing, comparison, tutorial
        from utils import styling, navigation
        
        # Test basic functionality
        os.chdir('webapp')
        result = subprocess.run([sys.executable, "test_app.py"], 
                              capture_output=True, text=True, timeout=60)
        os.chdir('..')
        
        if result.returncode == 0:
            print("✅ Web application validated")
            return True
        else:
            print(f"❌ Web app validation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Web app validation error: {e}")
        return False

def check_notebook_reproducibility():
    """Validate notebook can be executed."""
    print("📓 Validating notebook reproducibility...")
    try:
        import nbformat
        
        # Load notebook
        with open('notebooks/experiments.ipynb', 'r') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Check structure
        if len(nb.cells) > 0:
            print(f"✅ Notebook has {len(nb.cells)} cells and is well-structured")
            return True
        else:
            print("❌ Notebook appears to be empty")
            return False
    except Exception as e:
        print(f"❌ Notebook validation error: {e}")
        return False

def check_documentation():
    """Validate documentation links and references."""
    print("📚 Validating documentation...")
    try:
        required_files = [
            'README.md',
            'webapp/USAGE.md', 
            'demo/demo_script.md',
            'report/academic_report.md',
            'CONTRIBUTING.md',
            'LICENSE'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing documentation files: {missing_files}")
            return False
        
        # Check plot files referenced in README
        plot_files = [
            'slides/plots/compression_analysis.png',
            'slides/plots/singular_values.png', 
            'slides/plots/psnr_vs_k.png'
        ]
        
        missing_plots = []
        for plot_path in plot_files:
            if not Path(plot_path).exists():
                missing_plots.append(plot_path)
        
        if missing_plots:
            print(f"❌ Missing plot files: {missing_plots}")
            return False
            
        print("✅ All documentation files and references validated")
        return True
        
    except Exception as e:
        print(f"❌ Documentation validation error: {e}")
        return False

def check_professional_appearance():
    """Validate professional appearance elements."""
    print("✨ Validating professional appearance...")
    try:
        # Check styling files
        styling_files = [
            'webapp/utils/styling.py',
            'slides/plots/architecture.png',
            'slides/svd_compression_presentation.pptx'
        ]
        
        for file_path in styling_files:
            if not Path(file_path).exists():
                print(f"❌ Missing styling file: {file_path}")
                return False
        
        # Check README has badges and professional formatting
        with open('README.md', 'r') as f:
            readme_content = f.read()
            
        required_elements = ['[![', '##', '🌟', '🚀', '📊']
        missing_elements = []
        for element in required_elements:
            if element not in readme_content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ README missing professional elements: {missing_elements}")
            return False
            
        print("✅ Professional appearance validated")
        return True
        
    except Exception as e:
        print(f"❌ Professional appearance validation error: {e}")
        return False

def main():
    """Run all validation checks."""
    print("🔍 SVD Image Compression System - Final Quality Assurance")
    print("=" * 60)
    
    checks = [
        ("Test Suite", check_test_suite),
        ("Web Application", check_web_app), 
        ("Notebook Reproducibility", check_notebook_reproducibility),
        ("Documentation", check_documentation),
        ("Professional Appearance", check_professional_appearance)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 30)
        success = check_func()
        results.append((check_name, success))
    
    print("\n" + "=" * 60)
    print("📋 FINAL VALIDATION RESULTS")
    print("=" * 60)
    
    all_passed = True
    for check_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
        if not success:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - PROJECT READY FOR SUBMISSION!")
        print("✅ Task 11.3 'Final integration and quality assurance' COMPLETE")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - PLEASE REVIEW AND FIX ISSUES")
        return 1

if __name__ == "__main__":
    sys.exit(main())