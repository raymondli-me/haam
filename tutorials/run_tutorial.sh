#!/bin/bash
#
# HAAM Tutorial Runner
# ====================
# Runs HAAM tutorial scripts in sequence
#

echo "========================================================================"
echo "HAAM TUTORIAL RUNNER"
echo "========================================================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3."
    exit 1
fi

# Check if we need to install HAAM
if ! python3 -c "import haam" &> /dev/null; then
    echo "ERROR: HAAM not found."
    echo ""
    echo "Please install HAAM first:"
    echo "  cd .."
    echo "  pip install -e . sentence-transformers"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Navigate to tutorial directory
cd "$(dirname "$0")"

echo "========================================================================"
echo "TUTORIAL 1: Basic Example (Prestige only)"
echo "========================================================================"
echo ""
python3 01_basic_example.py

echo ""
echo "========================================================================"
echo "Press Enter to run full analysis (3 constructs)..."
echo "========================================================================"
read -r

echo "========================================================================"
echo "TUTORIAL 2: Full Analysis (Prestige, Power, Dominance)"
echo "========================================================================"
echo ""
python3 02_full_analysis.py

echo ""
echo "========================================================================"
echo "Press Enter to run with visualizations (wordclouds, UMAP, topics)..."
echo "WARNING: This will take longer and generate many files (~300 images)"
echo "========================================================================"
read -r

echo "========================================================================"
echo "TUTORIAL 3: Full Analysis with Visualizations"
echo "========================================================================"
echo ""
python3 03_with_visualizations.py

echo ""
echo "========================================================================"
echo "✓ ALL TUTORIALS COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  • Review the output above"
echo "  • Check haam_results_* folder for visualizations"
echo "  • Modify scripts to use your own data"
echo "  • See README.md for more details"
echo ""
