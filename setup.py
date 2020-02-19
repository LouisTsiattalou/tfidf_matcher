#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="tfidf_matcher",
    version="0.1.0",
    author="Louis Tsiattalou",
    author_email="louis.tsi@gmail.com",
    description="A small package that enables super-fast TF-IDF based string matching.",
    long_description="TODO",
    long_description_content_type="text/markdown",
    url="https://github.com/louistsiattalou/tfidf_matcher",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires='>=3.0',
    install_requires=["sklearn", "pandas"],
    setup_requires=["sklearn", "pandas"]
)
