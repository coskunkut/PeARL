from setuptools import setup, find_packages

setup(
    name="persistence",
    version="1.0",
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "statsmodels",
        "matplotlib",
        "tqdm",
        "tabulate"
    ],
    packages=find_packages()
)
