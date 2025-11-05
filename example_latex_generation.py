"""
Example: Generate LaTeX/TikZ Diagrams from HAAM Results
========================================================

This script demonstrates how to use the new `create_latex_diagram()` method
to generate publication-ready LaTeX/TikZ diagrams from HAAM analysis results.
"""

from haam import HAAM
import pandas as pd
import numpy as np

# Example: Load a saved HAAM model and generate LaTeX diagram
def generate_latex_from_model(model_path: str, trait_name: str, output_dir: str = "./latex_output"):
    """
    Load a saved HAAM model and generate LaTeX diagram.

    Parameters
    ----------
    model_path : str
        Path to saved .pkl model file
    trait_name : str
        Name of the trait (e.g., "Extraversion")
    output_dir : str
        Directory to save the LaTeX file
    """

    # Load the saved model
    haam_model = HAAM.load(model_path)

    # Generate LaTeX diagram with default settings (top 15 PCs, tri-sum ranking)
    result = haam_model.create_latex_diagram(
        trait_name=trait_name,
        output_dir=output_dir,
        n_pcs=15,              # Number of PCs to show (default: 15)
        coef_threshold=0.05,   # Show "--" for |coef| < 0.05
        render_pdf=True,       # Attempt to compile to PDF
        display=True           # Print status messages
    )

    print(f"\n✓ LaTeX diagram generated!")
    print(f"  - .tex file: {result['tex_path']}")
    if 'pdf_path' in result:
        print(f"  - .pdf file: {result['pdf_path']}")

    return result


# Example: Generate diagrams for all Big Five traits
def generate_all_traits(base_dir: str = "./HAAM_FULL_VIZ_20251002_030543"):
    """
    Generate LaTeX diagrams for all 5 Big Five traits from saved models.

    Parameters
    ----------
    base_dir : str
        Base directory containing trait model subdirectories
    """

    traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional_Stability",
        "Openness"
    ]

    results = {}

    for trait in traits:
        # Find the model file for this trait
        model_path = f"{base_dir}/models/{trait}_*/haam_model.pkl"

        # Use glob to find the actual path
        import glob
        matching_paths = glob.glob(model_path)

        if matching_paths:
            model_path = matching_paths[0]
            print(f"\n{'='*60}")
            print(f"Generating LaTeX diagram for {trait}")
            print(f"{'='*60}")

            output_dir = f"{base_dir}/latex_diagrams"
            result = generate_latex_from_model(model_path, trait, output_dir)
            results[trait] = result
        else:
            print(f"⚠ Model not found for {trait}")

    return results


# Example: Generate diagram with custom settings
def generate_custom_diagram(
    model_path: str,
    trait_name: str,
    n_pcs: int = 20,         # Show top 20 PCs instead of 15
    coef_threshold: float = 0.10,  # Higher threshold for showing coefficients
    render_pdf: bool = False
):
    """
    Generate diagram with custom settings.
    """

    haam_model = HAAM.load(model_path)

    result = haam_model.create_latex_diagram(
        trait_name=trait_name,
        output_dir="./custom_latex",
        n_pcs=n_pcs,
        coef_threshold=coef_threshold,
        render_pdf=render_pdf,
        display=True
    )

    return result


if __name__ == "__main__":
    print(__doc__)

    # Example 1: Generate for a single trait
    print("\n" + "="*60)
    print("EXAMPLE 1: Generate diagram for Extraversion")
    print("="*60)

    # Uncomment and modify path to your model file:
    # result = generate_latex_from_model(
    #     model_path="./HAAM_FULL_VIZ_20251002_030543/models/Extraversion_20251002_030543/haam_model.pkl",
    #     trait_name="Extraversion",
    #     output_dir="./latex_output"
    # )

    # Example 2: Generate for all traits
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate diagrams for all Big Five traits")
    print("="*60)

    # Uncomment and modify path to your base directory:
    # results = generate_all_traits(base_dir="./HAAM_FULL_VIZ_20251002_030543")

    # Example 3: Custom settings
    print("\n" + "="*60)
    print("EXAMPLE 3: Generate diagram with custom settings")
    print("="*60)

    # Uncomment and modify:
    # result = generate_custom_diagram(
    #     model_path="./path/to/model.pkl",
    #     trait_name="Custom Trait",
    #     n_pcs=20,
    #     coef_threshold=0.10,
    #     render_pdf=False
    # )

    print("\n" + "="*60)
    print("Examples shown above. Uncomment to run.")
    print("="*60)
