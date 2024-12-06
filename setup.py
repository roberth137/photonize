#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 20:57:11 2024

@author: roberthollmann
"""

from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("photon_sorter.pyx"),
    include_dirs=[np.get_include()],
)