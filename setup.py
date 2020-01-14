#!/usr/bin/env python

import setuptools

setuptools.setup(
    name="tfidf_matcher-louistsiattalou",
    version="0.0.1",
    author="Louis Tsiattalou",
    author_email="louis.tsi@gmail.com",
    description="A small package that enables super-fast TF-IDF based matching.",
    long_description="TODO",
#    install_requires=['pandas','numpy','re','sklearn'],
    long_description_content_type="text/markdown",
    url="https://github.com/louistsiattalou/tfidf_matcher",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)
