#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 21:40:52 2024

@author: roberthollmann
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("undrift.pyx"),
    include_dirs=[np.get_include()],
)