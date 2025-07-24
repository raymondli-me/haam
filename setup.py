#!/usr/bin/env python
"""Setup script for HAAM package."""

from setuptools import setup, find_packages
import os

# Try to read README, but don't fail if it doesn't exist
long_description = ""
if os.path.exists("haam_readme.md"):
    with open("haam_readme.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Human-AI Accuracy Model (HAAM) - Analyze how humans and AI use information"

setup(
    name="haam",
    version="0.1.0",
    author="Raymond Li",
    author_email="raymondli-me@github.com",
    description="Human-AI Accuracy Model (HAAM) - Analyze how humans and AI use information",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raymondli-me/haam",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "scikit-learn>=0.24.0",
        "statsmodels>=0.12.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "sentence-transformers>=2.0.0",
        "umap-learn>=0.5.0",
        "hdbscan>=0.8.27",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "haam=haam.cli:main",
        ],
    },
)