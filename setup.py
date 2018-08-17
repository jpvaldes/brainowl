#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup

# meta-data
NAME = 'brainowl'
DESCRIPTION = 'Classifier tuned for neuroimaging based on SpaRSA solver'
URL = 'https://github.com/jpvaldes/brainowl'
DOWNLOAD_URL = 'https://github.com/jpvaldes/brainowl.git'
EMAIL = 'jpvaldesherrera@gmail.com'
AUTHOR = 'Jose P Valdes Herrera'
REQUIRES_PYTHON = '>=3.5.0'

setup(
    name=NAME,
    version='0.1dev',
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=['brainowl'],
    license='BSD (3-clause)',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    download_url=DOWNLOAD_URL,
)
