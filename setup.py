#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 23:16:08 2024

@author: roberthollmann
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("histogram.pyx"),
    include_dirs=[np.get_include()],
)
