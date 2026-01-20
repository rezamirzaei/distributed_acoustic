#!/bin/bash
# =============================================================================
# Compile LaTeX Report to PDF (hardened)
# =============================================================================
# - Fails fast on common authoring errors (missing \end{document}, missing TeX)
# - Captures logs for troubleshooting
# - Prints the log tail and common error markers on failure
# =============================================================================

set -euo pipefail

# Change to the directory where this script is located
cd "$(dirname "$0")"

TEX_FILE="das_co2_monitoring_report.tex"
PDF_FILE="das_co2_monitoring_report.pdf"
LOG_FILE="${TEX_FILE%.tex}.log"

# Preconditions
if ! command -v pdflatex >/dev/null 2>&1; then
    echo "Error: 'pdflatex' not found in PATH. Install a TeX distribution (e.g., MacTeX)." >&2
    exit 1
fi

if [ ! -f "$TEX_FILE" ]; then
    echo "Error: $TEX_FILE not found in $(pwd)." >&2
    exit 1
fi

# Fast authoring sanity checks (prevents confusing LaTeX stops later)
if ! grep -F "\end{document}" "$TEX_FILE" >/dev/null; then
    echo "Error: $TEX_FILE does not contain \\end{document}." >&2
    echo "Hint: The file may have been truncated or is missing its final lines." >&2
    exit 1
fi

echo "=============================================="
echo "  Compiling LaTeX Report to PDF"
echo "=============================================="
echo ""

echo "[1/3] First pass..."
# Keep logs for later inspection but avoid spewing full output.
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >"${LOG_FILE}.pass1.stdout" 2>"${LOG_FILE}.pass1.stderr" || {
    echo "" >&2
    echo "==============================================" >&2
    echo "  ERROR: pdflatex failed on pass 1" >&2
    echo "==============================================" >&2
    [ -f "$LOG_FILE" ] && { echo "--- Log tail ($LOG_FILE) ---" >&2; tail -n 60 "$LOG_FILE" >&2; }
    exit 1
}

echo "[2/3] Second pass (for TOC and references)..."
pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE" >"${LOG_FILE}.pass2.stdout" 2>"${LOG_FILE}.pass2.stderr" || {
    echo "" >&2
    echo "==============================================" >&2
    echo "  ERROR: pdflatex failed on pass 2" >&2
    echo "==============================================" >&2
    [ -f "$LOG_FILE" ] && { echo "--- Log tail ($LOG_FILE) ---" >&2; tail -n 60 "$LOG_FILE" >&2; }
    exit 1
}

echo "[3/3] Third pass (final)..."
# Third pass: show output interactively to help diagnose warnings quickly.
if ! pdflatex -interaction=nonstopmode -halt-on-error "$TEX_FILE"; then
    echo "" >&2
    echo "==============================================" >&2
    echo "  ERROR: PDF generation failed on pass 3" >&2
    echo "==============================================" >&2

    if [ -f "$LOG_FILE" ]; then
        echo "--- Log tail ($LOG_FILE) ---" >&2
        tail -n 80 "$LOG_FILE" >&2

        echo "" >&2
        echo "--- Common error markers (first matches) ---" >&2
        # Try to surface the most relevant lines.
        grep -nE "^! |Fatal error|Emergency stop|Undefined control sequence|LaTeX Error" "$LOG_FILE" | head -n 20 >&2 || true
    fi

    exit 1
fi

# Verify output
if [ ! -f "$PDF_FILE" ]; then
    echo "" >&2
    echo "==============================================" >&2
    echo "  ERROR: pdflatex reported success but $PDF_FILE was not created." >&2
    echo "==============================================" >&2
    [ -f "$LOG_FILE" ] && { echo "--- Log tail ($LOG_FILE) ---" >&2; tail -n 80 "$LOG_FILE" >&2; }
    exit 1
fi

echo ""
echo "=============================================="
echo "  SUCCESS! PDF generated: $PDF_FILE"
echo "=============================================="
echo ""

# Clean up auxiliary files (keep the pass stdout/stderr for debugging)
echo "Cleaning up auxiliary files..."
rm -f *.aux *.toc *.out *.bbl *.blg *.lof *.lot *.fls *.fdb_latexmk
# Keep the main .log by default (useful for warnings); delete only if user wants.
# rm -f "$LOG_FILE"
echo "Done."

# Open the PDF (macOS)
if [[ "${OSTYPE:-}" == "darwin"* ]]; then
    echo ""
    echo "Opening PDF..."
    open "$PDF_FILE"
fi
