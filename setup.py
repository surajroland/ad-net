#!/usr/bin/env python3
"""Setup script for ADNet package"""

import tomllib
from pathlib import Path
from typing import Dict, List

from setuptools import find_packages, setup

# Read README for long description
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text()
except FileNotFoundError:
    long_description = (
        "4D spatio-temporal object detection framework for advanced driving"
    )


# Read metadata from pyproject.toml
def get_project_metadata() -> Dict[str, str]:
    """Read project metadata from pyproject.toml"""

    with open("pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    project = pyproject.get("project", {})
    poetry = pyproject.get("tool", {}).get("poetry", {})

    return {
        "version": project.get("version") or poetry["version"],
        "description": project.get("description") or poetry["description"],
        "author": (
            project.get("authors", [{}])[0].get("name")
            if project.get("authors")
            else poetry["authors"][0].split(" <")[0]
        ),
        "author_email": (
            project.get("authors", [{}])[0].get("email")
            if project.get("authors")
            else poetry["authors"][0].split(" <")[1].rstrip(">")
        ),
        "url": project.get("urls", {}).get("Repository") or poetry["repository"],
    }


# Helper function to read requirements files
def read_requirements(filename: str) -> List[str]:
    """Read requirements from requirements/ directory"""
    requirements_path = this_directory / "requirements" / filename
    if requirements_path.exists():
        lines = requirements_path.read_text().strip().split("\n")
        # Filter out comments and empty lines
        return [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
    return []


metadata = get_project_metadata()

setup(
    # Package metadata
    name="adnet",
    version=metadata["version"],
    author=metadata["author"],
    author_email=metadata["author_email"],
    description=metadata["description"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=metadata["url"],
    # Package structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # Classification
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: POSIX :: Linux",
    ],
    # Requirements
    python_requires=">=3.12",
    install_requires=read_requirements("base.txt"),
    extras_require={
        "dev": read_requirements("dev.txt"),
        "training": read_requirements("training.txt"),
        "api": read_requirements("api.txt"),
        "visualization": read_requirements("visualization.txt"),
    },
    # CLI entry points
    entry_points={
        "console_scripts": [
            "adnet=adnet.cli:main",
        ],
    },
    # Package configuration
    include_package_data=True,
    zip_safe=False,
)
