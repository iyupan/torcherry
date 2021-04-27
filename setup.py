# -*- coding: utf-8 -*-

from torcherry import __version__, __author__
from setuptools import setup, find_packages

setup(
      name='torcherry',
      version=__version__,
      description='torcherry: a higher framework for pytorch.',
      author=__author__,
      maintainer=__author__,
      url='https://github.com/perryuu/torcherry',
      packages=find_packages(),
      py_modules=[],
      long_description="Make training models more easy.",
      license="GPLv3",
      platforms=["any"],
      install_requires = ["torch>=1.0.0"]
)
