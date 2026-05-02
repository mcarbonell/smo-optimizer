from setuptools import setup, find_packages

setup(
    name="supermario-optimizer",
    version="0.1.0",
    description="Super Mario Optimizer (SMO) - An ultra-memory-efficient PyTorch optimizer using spatial and spectral compression.",
    author="Mario",
    url="https://github.com/mcarbonell/supermario-optimizer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
