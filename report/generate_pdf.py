#!/usr/bin/env python3
"""
Script to generate PDF report from LaTeX source.
Handles LaTeX compilation with proper error handling and cleanup.
"""

import subprocess
import sys
from pathlib import Path
import shutil
import os

def check_latex_installation():
    """Check if LaTeX is installed and available."""
    try:
        result = subprocess.run(['pdflatex', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ LaTeX installation found")
            return True
        else:
            print("❌ LaTeX not found or not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ LaTeX (pdflatex) not found in PATH")
        return False

def compile_latex(tex_file: Path, output_dir: Path = None):
    """
    Compile LaTeX file to PDF.
    
    Args:
        tex_file: Path to .tex file
        output_dir: Directory for output files (default: same as tex_file)
    
    Returns:
        bool: True if compilation successful, False otherwise
    """
    if output_dir is None:
        output_dir = tex_file.parent
    
    # Change to the directory containing the tex file
    original_cwd = os.getcwd()
    os.chdir(tex_file.parent)
    
    try:
        print(f"📝 Compiling {tex_file.name}...")
        
        # First pass
        result1 = subprocess.run([
            'pdflatex', 
            '-interaction=nonstopmode',
            '-output-directory', str(output_dir),
            tex_file.name
        ], capture_output=True, text=True, timeout=60)
        
        if result1.returncode != 0:
            print("❌ First LaTeX pass failed:")
            print(result1.stdout[-1000:])  # Last 1000 chars of output
            return False
        
        print("✅ First pass completed")
        
        # Second pass (for references, TOC, etc.)
        result2 = subprocess.run([
            'pdflatex', 
            '-interaction=nonstopmode',
            '-output-directory', str(output_dir),
            tex_file.name
        ], capture_output=True, text=True, timeout=60)
        
        if result2.returncode != 0:
            print("⚠️ Second LaTeX pass had issues, but PDF may still be generated")
            print(result2.stdout[-500:])  # Last 500 chars of output
        else:
            print("✅ Second pass completed")
        
        # Check if PDF was generated
        pdf_file = output_dir / tex_file.with_suffix('.pdf').name
        if pdf_file.exists():
            print(f"🎉 PDF generated successfully: {pdf_file}")
            return True
        else:
            print("❌ PDF file not found after compilation")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ LaTeX compilation timed out")
        return False
    except Exception as e:
        print(f"❌ Error during compilation: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def cleanup_latex_files(tex_file: Path):
    """Clean up auxiliary LaTeX files."""
    aux_extensions = ['.aux', '.log', '.toc', '.out', '.fls', '.fdb_latexmk', '.synctex.gz']
    
    for ext in aux_extensions:
        aux_file = tex_file.with_suffix(ext)
        if aux_file.exists():
            try:
                aux_file.unlink()
                print(f"🧹 Cleaned up {aux_file.name}")
            except Exception as e:
                print(f"⚠️ Could not remove {aux_file.name}: {e}")

def generate_markdown_pdf_fallback(md_file: Path):
    """
    Generate PDF from Markdown using pandoc as fallback.
    
    Args:
        md_file: Path to markdown file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if pandoc is available
        subprocess.run(['pandoc', '--version'], 
                      capture_output=True, timeout=5)
        
        pdf_file = md_file.with_suffix('.pdf')
        
        print(f"📝 Converting {md_file.name} to PDF using pandoc...")
        
        result = subprocess.run([
            'pandoc',
            str(md_file),
            '-o', str(pdf_file),
            '--pdf-engine=pdflatex',
            '--variable', 'geometry:margin=1in',
            '--variable', 'fontsize=12pt',
            '--variable', 'documentclass=article',
            '--number-sections',
            '--toc'
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0 and pdf_file.exists():
            print(f"🎉 PDF generated successfully: {pdf_file}")
            return True
        else:
            print("❌ Pandoc conversion failed:")
            print(result.stderr)
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Pandoc not found - cannot generate PDF from Markdown")
        return False

def main():
    """Main function to generate PDF report."""
    print("🚀 Starting PDF generation process...")
    
    # Get the report directory
    report_dir = Path(__file__).parent
    tex_file = report_dir / 'academic_report.tex'
    md_file = report_dir / 'academic_report.md'
    
    # Check if files exist
    if not tex_file.exists():
        print(f"❌ LaTeX file not found: {tex_file}")
        if md_file.exists():
            print("📄 Markdown file found, attempting pandoc conversion...")
            success = generate_markdown_pdf_fallback(md_file)
            sys.exit(0 if success else 1)
        else:
            print(f"❌ No source files found in {report_dir}")
            sys.exit(1)
    
    # Check LaTeX installation
    if not check_latex_installation():
        print("\n💡 LaTeX not found. Trying pandoc fallback...")
        if md_file.exists():
            success = generate_markdown_pdf_fallback(md_file)
            sys.exit(0 if success else 1)
        else:
            print("\n❌ Neither LaTeX nor pandoc available for PDF generation")
            print("Please install LaTeX (e.g., TeX Live) or pandoc to generate PDFs")
            sys.exit(1)
    
    # Compile LaTeX to PDF
    success = compile_latex(tex_file)
    
    if success:
        print("\n🎉 PDF generation completed successfully!")
        
        # Optional cleanup
        cleanup_choice = input("\n🧹 Clean up auxiliary LaTeX files? (y/N): ").strip().lower()
        if cleanup_choice in ['y', 'yes']:
            cleanup_latex_files(tex_file)
        
        print(f"\n📄 Your report is ready: {tex_file.with_suffix('.pdf')}")
    else:
        print("\n❌ PDF generation failed")
        print("\n💡 Trying pandoc fallback...")
        if md_file.exists():
            success = generate_markdown_pdf_fallback(md_file)
            if not success:
                print("\n❌ All PDF generation methods failed")
                sys.exit(1)
        else:
            sys.exit(1)

if __name__ == '__main__':
    main()