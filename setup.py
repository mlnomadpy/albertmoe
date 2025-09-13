"""
Setup script for AlbertMoE package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="albertmoe",
    version="0.1.0",
    author="MLNomadPy",
    author_email="",
    description="A modular implementation of ALBERT with Mixture of Experts",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mlnomadpy/albertmoe",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "isort>=5.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "albertmoe-train-clm=scripts.train_clm:main",
            "albertmoe-train-mlm=scripts.train_mlm:main",
        ],
    },
    include_package_data=True,
    package_data={
        "albertmoe": ["*.py"],
    },
)