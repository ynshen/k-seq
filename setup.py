#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='k-seq',
    version='0.0-dev1',
    description='Package for kinetic analysis, Chen lab at University of California, Santa Barbara',
    url='https://github.com/ynshen/k-seq',
    author='Yuning Shen',
    author_email='ynshen23@gmail.com',
    packages=find_packages('src/pkg'),
    package_dir={'': 'src/pkg'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'python-levenshtein',
        'scipy',
        'seaborn',
        'jupyter',
	'dill',
	'jsonlib-python3'
    ],
    zip_safe=False
)
