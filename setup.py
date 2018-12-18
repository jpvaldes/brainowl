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

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    name=NAME,
    version='0.1dev',
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=['brainowl'],
    license='BSD (3-clause)',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    download_url=DOWNLOAD_URL,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
    ],
)
