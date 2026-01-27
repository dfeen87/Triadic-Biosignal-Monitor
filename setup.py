"""
Operator-Based Heart-Brain Monitoring Framework
Setup script for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements = []
with open("requirements.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Skip comments and empty lines
        if line and not line.startswith("#"):
            # Remove version constraints for setup.py (pip will handle)
            package = line.split(">=")[0].split("==")[0].split("<")[0]
            requirements.append(package)

# Development requirements
dev_requirements = [
    "pytest>=6.2.0",
    "pytest-cov>=2.12.0",
    "black>=21.0",
    "flake8>=3.9.0",
    "pylint>=2.10.0",
    "mypy>=0.910",
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=0.5.0",
]

setup(
    name="triadic-biosignal-monitor",
    version="1.0.0",
    author="Marcel Krüger, Don Feeney",
    author_email="marcelkrueger092@gmail.com, dfeen87@gmail.com",
    description="Operator-based heart-brain monitoring via triadic spiral-time embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dfeen87/Triadic-Biosignal-Monitor",
    project_urls={
        "Bug Tracker": "https://github.com/dfeen87/Triadic-Biosignal-Monitor/issues",
        "Documentation": "https://github.com/dfeen87/Triadic-Biosignal-Monitor/tree/main/docs",
        "Source Code": "https://github.com/dfeen87/Triadic-Biosignal-Monitor",
    },
    packages=find_packages(where=".", exclude=["tests", "tests.*", "notebooks", "datasets"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "License :: OSI Approved :: MIT License",  # Update if different license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "eeg": [
            "mne>=1.0.0",  # For EEG data loading
        ],
        "all": dev_requirements + ["mne>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            # Add command-line scripts here if needed
            # "triadic-monitor=core.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.yaml", "*.yml"],
    },
    zip_safe=False,
    keywords=[
        "biosignal monitoring",
        "EEG",
        "ECG",
        "HRV",
        "seizure detection",
        "arrhythmia detection",
        "early warning system",
        "signal processing",
        "operator theory",
        "phase synchronization",
    ],
)
