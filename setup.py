#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name='k-seq',
    version='0.3.0-dev',
    description='Package for kinetic sequencing analysis, Chen lab at University of California, Santa Barbara',
    url='https://github.com/ynshen/k-seq',
    author='Yuning Shen',
    author_email='ynshen23@gmail.com',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'python-levenshtein',
        'scipy',
        'seaborn',
        'jupyter',
	'jsonlib-python3',
    ],
    zip_safe=False
)
