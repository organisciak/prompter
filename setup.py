#!/usr/bin/env python

from distutils.core import setup

setup(name='Prompter',
      version='0.1.0',
      description='Text-to-Speech Prompt Helper',
      author='Peter Organisciak',
      author_email='organisciak@gmail.com',
      url='https://www.porganized.com',
      requires=['pandas', 'numpy', 'ipywidgets', 'numpy'],
      packages=['prompter']
     )