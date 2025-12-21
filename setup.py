"""
Setup script for RAJNI-ViT package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="rajni-vit",
    version="0.1.0",
    author="Dhairya Raniwal",
    description="RAJNI: Relative Adaptive Jacobian-based Neuronal Importance for Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dRaniwal/RAJNI-ViT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "full": [
            "timm>=0.9.0",
            "pillow>=9.0.0",
            "tqdm>=4.65.0",
        ],
    },
)
