"""Setup file

This file follows https://packaging.python.org/en/latest/distributing.html
"""

from os import path
from setuptools import setup

# meta-data
NAME = 'brainowl'
DESCRIPTION = 'Classifier tuned for neuroimaging based on SpaRSA solver'
URL = 'https://github.com/jpvaldes/brainowl'
DOWNLOAD_URL = 'https://github.com/jpvaldes/brainowl.git'
EMAIL = 'jpvaldesherrera@gmail.com'
AUTHOR = 'Jose P Valdes Herrera'
REQUIRES_PYTHON = '>=3.5.0'
REQUIRES_PACKAGES = ['numpy', 'scipy', 'scikit-learn']

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, 'README.md'), 'r') as readme_file:
    LONG_DESCRIPTION = readme_file.read()

setup(
    name=NAME,
    version='0.1',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=[NAME],
    license='BSD (3-clause)',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    install_requires=REQUIRES_PACKAGES,
    url=URL,
    download_url=DOWNLOAD_URL,
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ],
)
