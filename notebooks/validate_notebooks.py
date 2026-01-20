#!/usr/bin/env python3
"""Validate all notebooks in the directory"""
import json
import os
import glob

def validate_notebooks():
    notebook_dir = os.path.dirname(os.path.abspath(__file__))
    notebooks = glob.glob(os.path.join(notebook_dir, '*.ipynb'))

    print(f"Found {len(notebooks)} notebooks\n")

    for nb_path in sorted(notebooks):
        name = os.path.basename(nb_path)
        try:
            with open(nb_path, 'r') as f:
                nb = json.load(f)

            cells = len(nb.get('cells', []))
            nbformat = nb.get('nbformat', 'unknown')

            # Count cell types
            code_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'code')
            md_cells = sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')

            print(f"✅ {name}")
            print(f"   Cells: {cells} ({code_cells} code, {md_cells} markdown)")
            print(f"   nbformat: {nbformat}")

        except json.JSONDecodeError as e:
            print(f"❌ {name} - INVALID JSON: {e}")
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
        print()

if __name__ == '__main__':
    validate_notebooks()
