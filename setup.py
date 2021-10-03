"""PRML setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject/blob/main/setup.py
"""

from setuptools import find_packages, setup


setup(
    name="prml",
    version="0.0.1",
    description="Collection of PRML algorithms",
    author="ctgk",
    python_requires=">=3.6",
    install_requires=["numpy", "scipy"],
    packages=find_packages(exclude=["test", "test.*"]),
    test_suite="test",
)
