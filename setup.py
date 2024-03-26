# -*- coding: utf-8 -*-

from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='OpenCOOD_LRCP',
    version=__version__,
    packages=find_packages(),
    license='MIT',
    description='An opensource pytorch framework for autonomous driving '
                'cooperative detection',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)
