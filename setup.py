"""
Setup script for RAJNI-ViT.

RAJNI: Relative Adaptive Jacobian-based Neuronal Importance
for efficient Vision Transformers via adaptive token pruning.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rajni-vit",
    version="0.1.0",
    author="Dhairya Raniwal",
    author_email="",
    description="Adaptive Jacobian-based Token Pruning for Vision Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dRaniwal/RAJNI-ViT",
    packages=find_packages(exclude=["tests", "examples", "scripts"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "timm>=0.9.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
)
