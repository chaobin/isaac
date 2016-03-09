# coding: utf-8
from __future__ import print_function
import os
from os.path import join as pjoin
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install as _install


PROJECT_NAME = 'isaac'


setup(
    name=PROJECT_NAME,
    version='1.0',
    packages=find_packages(),
    long_description='A machine learning library',
    install_requires=[
        "numpy",
    ]
)
