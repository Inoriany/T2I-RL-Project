"""Setup script for T2I-RL package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="t2i-rl",
    version="0.1.0",
    author="T2I-RL Team",
    author_email="",
    description="Text-to-Image Generation with Reinforcement Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Inoriany/T2I-RL-Project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "t2i-train=scripts.train:main",
            "t2i-eval=scripts.evaluate:main",
        ],
    },
)
