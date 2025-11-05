"""
Example: Generate LaTeX Tables from HAAM Results
=================================================

This script demonstrates how to use the LaTeX table generation methods
to create publication-ready tables from HAAM analysis results.
"""

from haam import HAAM

# Example 1: Generate all tables at once
def generate_all_tables_example(model_path: str, trait_name: str):
    """Generate all 5 tables for a trait."""

    # Load the HAAM model
    haam_model = HAAM.load(model_path)

    # Generate all tables at once
    results = haam_model.create_all_latex_tables(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    print("\n" + "="*60)
    print("Generated Tables:")
    print("="*60)
    for table_name, result in results.items():
        print(f"{table_name}: {result['tex_path']}")

    return results


# Example 2: Generate individual tables
def generate_individual_tables_example(model_path: str, trait_name: str):
    """Generate tables one by one for debugging."""

    haam_model = HAAM.load(model_path)

    # Table 1: Zero-Order Correlations
    table1 = haam_model.create_table_zero_order_correlations(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    # Table 2: LASSO Feature Selection
    table2 = haam_model.create_table_lasso_selection(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    # Table 3: R² and PoMA
    table3 = haam_model.create_table_r2_and_poma(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    # Table 4: DML Effects
    table4 = haam_model.create_table_dml_effects(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    # Table 7: G and C Parameters
    table7 = haam_model.create_table_g_and_c(
        trait_name=trait_name,
        output_dir="./latex_tables",
        display=True
    )

    return {
        'table1': table1,
        'table2': table2,
        'table3': table3,
        'table4': table4,
        'table7': table7
    }


# Example 3: Batch generate for all Big Five traits
def generate_all_traits_tables(base_dir: str = "./HAAM_FULL_VIZ_20251002_030543"):
    """Generate all tables for all 5 Big Five traits."""

    traits = [
        "Extraversion",
        "Agreeableness",
        "Conscientiousness",
        "Emotional_Stability",
        "Openness"
    ]

    import glob
    all_results = {}

    for trait in traits:
        # Find the model file
        model_pattern = f"{base_dir}/models/{trait}_*/haam_model.pkl"
        matching_paths = glob.glob(model_pattern)

        if matching_paths:
            model_path = matching_paths[0]
            print(f"\n{'='*60}")
            print(f"Generating tables for {trait}")
            print(f"{'='*60}")

            results = generate_all_tables_example(model_path, trait)
            all_results[trait] = results
        else:
            print(f"⚠ Model not found for {trait}")

    return all_results


# Example 4: Custom output directory per table type
def organize_by_table_type(model_path: str, trait_name: str):
    """Save each table type to its own directory."""

    haam_model = HAAM.load(model_path)

    # Create organized structure
    structure = {
        'zero_order': './tables/zero_order',
        'lasso': './tables/lasso',
        'r2_poma': './tables/r2_poma',
        'dml': './tables/dml',
        'g_and_c': './tables/g_and_c'
    }

    results = {}

    results['table1'] = haam_model.create_table_zero_order_correlations(
        trait_name=trait_name,
        output_dir=structure['zero_order'],
        display=True
    )

    results['table2'] = haam_model.create_table_lasso_selection(
        trait_name=trait_name,
        output_dir=structure['lasso'],
        display=True
    )

    results['table3'] = haam_model.create_table_r2_and_poma(
        trait_name=trait_name,
        output_dir=structure['r2_poma'],
        display=True
    )

    results['table4'] = haam_model.create_table_dml_effects(
        trait_name=trait_name,
        output_dir=structure['dml'],
        display=True
    )

    results['table7'] = haam_model.create_table_g_and_c(
        trait_name=trait_name,
        output_dir=structure['g_and_c'],
        display=True
    )

    return results


if __name__ == "__main__":
    print(__doc__)

    # Example 1: Generate all tables for a single trait
    print("\n" + "="*60)
    print("EXAMPLE 1: Generate all tables at once")
    print("="*60)

    # Uncomment and modify path:
    # results = generate_all_tables_example(
    #     model_path="./HAAM_FULL_VIZ_20251002_030543/models/Extraversion_20251002_030543/haam_model.pkl",
    #     trait_name="Extraversion"
    # )

    # Example 2: Generate tables individually
    print("\n" + "="*60)
    print("EXAMPLE 2: Generate tables individually")
    print("="*60)

    # Uncomment and modify path:
    # results = generate_individual_tables_example(
    #     model_path="./path/to/model.pkl",
    #     trait_name="Power"
    # )

    # Example 3: Batch generate for all traits
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch generate for all Big Five traits")
    print("="*60)

    # Uncomment and modify path:
    # all_results = generate_all_traits_tables(
    #     base_dir="./HAAM_FULL_VIZ_20251002_030543"
    # )

    # Example 4: Organize by table type
    print("\n" + "="*60)
    print("EXAMPLE 4: Organize tables by type")
    print("="*60)

    # Uncomment and modify path:
    # results = organize_by_table_type(
    #     model_path="./path/to/model.pkl",
    #     trait_name="Dominance"
    # )

    print("\n" + "="*60)
    print("Examples shown above. Uncomment to run.")
    print("="*60)


# Quick reference for available methods:
"""
Available table generation methods:

1. create_table_zero_order_correlations(trait_name, output_dir, display)
   - Table 1: r correlations between Validity, AI, Human

2. create_table_lasso_selection(trait_name, output_dir, display)
   - Table 2: Number of PCs selected by LASSO

3. create_table_r2_and_poma(trait_name, output_dir, display)
   - Table 3: Cross-validated R² and PoMA values
   - Shows both CV and training set R²

4. create_table_dml_effects(trait_name, output_dir, display)
   - Table 4: Total (β), DML Direct (β̌), Indirect effects
   - Includes SE, t-statistics, p-values, 95% CI

5. create_table_g_and_c(trait_name, output_dir, display)
   - Table 7: Policy Similarity (G) and Residual Correlation (C)

6. create_all_latex_tables(trait_name, output_dir, display)
   - Generates all 5 tables at once
"""
