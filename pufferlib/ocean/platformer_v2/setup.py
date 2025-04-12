from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extension = Extension(
    "cy_platformer",
    sources=["cy_platformer.pyx", "platformer.c"],
    include_dirs=[np.get_include()],
    libraries=["raylib"],  # Link against raylib for rendering
)

setup(
    name="cy_platformer",
    ext_modules=cythonize([extension]),
) 