"""
Script to create all visualizations for the paper.
This script serves as a convenience wrapper around the
visualization classes to generate all required figures.
"""

import os
import numpy as np
from primality_test import CirulantMatrixPrimalityTest, CyclotomicVisualization
from cyclotomic_visualization import CyclotomicVisualizer

def main():
    # Create output directory
    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    # Set range of integers to analyze
    n_values = list(range(90, 130))

    print("Generating visualizations using both implementations...")

    # Method 1: Using improved_primality_test.py
    print("\nUsing CyclotomicVisualization from improved_primality_test.py")
    vis1 = CyclotomicVisualization()

    # Generate main plots
    vis1.create_visualization(n_values, save_path=f"{output_dir}/vis1_cyclotomic_visualization.pdf")

    # Method 2: Using cyclotomic_visualization.py
    print("\nUsing CyclotomicVisualizer from cyclotomic_visualization.py")
    vis2 = CyclotomicVisualizer(output_dir=output_dir)

    # Generate individual plots
    vis2.plot_factorization_patterns(n_values, filename="factorization_patterns.pdf")
    vis2.plot_eigenvalue_distributions(n_values, filename="eigenvalue_distributions.pdf")
    vis2.plot_field_extensions(n_values, filename="field_extensions.pdf")
    vis2.plot_coefficient_patterns(n_values, filename="coefficient_patterns.pdf")
    vis2.plot_dynamic_system(n_values, filename="dynamic_system.pdf")

    # Generate combined plots
    vis2.create_full_analysis(n_values, filename="full_analysis.pdf")
    vis2.create_comprehensive_visualization(n_values, filename="cyclotomic_visualization.pdf")

    print("\nAll visualizations generated successfully!")
    print(f"Output files saved to: {os.path.abspath(output_dir)}")

    # List all generated files
    print("\nGenerated files:")
    for filename in sorted(os.listdir(output_dir)):
        print(f"  - {filename}")


if __name__ == "__main__":
    main()