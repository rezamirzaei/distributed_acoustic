#!/bin/bash
# =============================================================================
# Compile LaTeX Report to PDF
# =============================================================================
# This script compiles das_co2_monitoring_report.tex to PDF using pdflatex.
# Run it from the report/ directory or use: ./compile_report.sh
# =============================================================================

# Change to the directory where this script is located
cd "$(dirname "$0")"

# Define file names
TEX_FILE="das_co2_monitoring_report.tex"
PDF_FILE="das_co2_monitoring_report.pdf"

# Check if the tex file exists
if [ ! -f "$TEX_FILE" ]; then
    echo "Error: $TEX_FILE not found in the current directory."
    exit 1
fi

echo "=============================================="
echo "  Compiling LaTeX Report to PDF"
echo "=============================================="
echo ""

# Run pdflatex multiple times for cross-references and TOC
echo "[1/3] First pass..."
pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1

echo "[2/3] Second pass (for TOC and references)..."
pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1

echo "[3/3] Third pass (final)..."
pdflatex -interaction=nonstopmode "$TEX_FILE"

# Check if PDF was created successfully
if [ -f "$PDF_FILE" ]; then
    echo ""
    echo "=============================================="
    echo "  SUCCESS! PDF generated: $PDF_FILE"
    echo "=============================================="
    echo ""

    # Clean up auxiliary files
    echo "Cleaning up auxiliary files..."
    rm -f *.aux *.log *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk
    echo "Done."

    # Open the PDF (macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo "Opening PDF..."
        open "$PDF_FILE"
    fi
else
    echo ""
    echo "=============================================="
    echo "  ERROR: PDF generation failed."
    echo "  Check the log file for details: ${TEX_FILE%.tex}.log"
    echo "=============================================="
    exit 1
fi
