#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='Prompter',
      version='0.2.2',
      description='Text-to-Speech Prompt Helper',
      author='Peter Organisciak',
      author_email='organisciak@gmail.com',
      url='https://www.porganized.com',
      requires=['pandas', 'smart_open', 'numpy', 'ipywidgets', 'numpy'],
      packages=['prompter'],
      include_package_data=True
     )