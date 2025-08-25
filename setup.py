from setuptools import setup, find_packages
from pathlib import Path

# Read the README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="ai-training-preflight",
    version="0.1.0",
    author="Krunal Patel",
    author_email="",  # Add your email if you want
    description="Predict AI training failures through static analysis before wasting compute resources",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kpatel513/ai-training-preflight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "click>=8.0.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=3.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "preflight=preflight.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt"],
        "examples": ["*.py"],
    },
    project_urls={
        "Bug Reports": "https://github.com/kpatel513/ai-training-preflight/issues",
        "Source": "https://github.com/kpatel513/ai-training-preflight",
    },
    keywords=[
        "deep learning",
        "machine learning", 
        "pytorch",
        "training",
        "debugging",
        "static analysis",
        "memory profiling",
        "gpu optimization",
        "ai",
        "neural networks"
    ],
)