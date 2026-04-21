from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import platform

# Configuración de flags
if platform.system() == "Windows":
    compile_args = ['/openmp', '/O2']
    link_args = []
else:
    compile_args = ['-fopenmp', '-O3']
    link_args = ['-fopenmp']

ext_modules = [
    Extension(
        "im2col", # <--- Cambia "im2col_cython" por "im2col"
        sources=["cython_modules/im2col.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    )
]

setup(
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={"language_level": "3"},
    ),
)