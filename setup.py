from setuptools import find_packages
from distutils.core import setup

setup(
    name='shape_oa',
    version='1.0.0',
    author='KAI',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='',
    description='Isaac Gym environments for SHAPE_OA',
    install_requires=['isaacgym',
                      'rsl-rl',
                      'matplotlib']
)