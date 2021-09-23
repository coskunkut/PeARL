from setuptools import setup, find_packages

setup(
    name="pbnm",
    version="1.0",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "statsmodels",
        "matplotlib",
        "tqdm",
        "more-itertools",
        "tabulate",
        "natsort",
        "filterpy",
        "bkmeans",
        "huffman"
    ],
    packages=find_packages()
)
