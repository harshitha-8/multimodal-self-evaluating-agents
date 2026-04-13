"""
Multimodal Self-Evaluating Agents (MSEA)
Research framework for multimodal reasoning, agent metacognition,
and annotation-efficient learning.
"""

from setuptools import setup, find_packages

setup(
    name="msea",
    version="0.1.0",
    author="Harshitha M",
    author_email="harshitha@example.com",
    description="Multimodal Self-Evaluating Agents: Research framework for agent metacognition",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/harshitha-8/multimodal-self-evaluating-agents",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "transformers>=4.36.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "einops>=0.7.0",
        "open-clip-torch>=2.23.0",
        "wandb>=0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
