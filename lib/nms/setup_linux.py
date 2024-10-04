# --------------------------------------------------------
# Pose.gluon
# Copyright (c) 2018-present Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


def find_in_path(name, path):
    """Find a file in a search path."""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system.
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib'.
    """
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc.exe')
    else:
        nvcc = find_in_path(os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib': pjoin(home, 'lib')}
    for k, v in cudaconfig.items():
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be located in %s' % (k, v))

    return cudaconfig


CUDA = locate_cuda()


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """Customize how the dispatch to nvcc works."""
    self.src_extensions.append('.cu')

    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_cxx', CUDA['nvcc'])  # Use compiler_cxx for nvcc
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super_compile(obj, src, ext, cc_args, postargs, pp_opts)

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={'gcc': ["/W3", "/wd4244"],  # Disable warning for possible data loss
                            'nvcc': []},
        include_dirs=[numpy_include]
    ),
    Extension('gpu_nms',
        ['nms_kernel.cu', 'gpu_nms.pyx'],
        library_dirs=[CUDA['lib']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib']],
        extra_compile_args={'gcc': ["/W3", "/wd4244"],  # Disable warning for possible data loss
                            'nvcc': ['/arch=sm_61',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     '/Zi']},
        include_dirs=[numpy_include, CUDA['include']]
    ),
]

setup(
    name='nms',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
