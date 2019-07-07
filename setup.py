from setuptools import setup, find_packages


setup(
    name="prml",
    version="0.1.0",
    description="Collection of PRML algorithms",
    author="ctgk",
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
    extras_require={
        "dev": [
            "flake8",
            "jupyter",
            "matplotlib",
            "scikit-learn",
            "sphinx",
            "sphinx_rtd_theme"
        ]
    },
    packages=find_packages(exclude=["test", "test.*"]),
    test_suite="test"
)
