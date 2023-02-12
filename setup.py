#!/usr/bin/env python

from setuptools import setup, find_packages
long_description_md = open("README.md", encoding = "utf-8").read()

setup(
    name="tfidf_matcher",
    version="0.2.0",
    author="Louis Tsiattalou",
    author_email="louis.tsi@gmail.com",
    description="A small package that enables super-fast TF-IDF based string matching.",
    long_description=long_description_md,
    long_description_content_type="text/markdown",
    url="https://github.com/louistsiattalou/tfidf_matcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.5',
    install_requires=["scikit-learn", "pandas"],
)
