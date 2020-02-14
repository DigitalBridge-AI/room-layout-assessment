"""
Setup file for room_layout_assessment.
"""

from setuptools import setup, find_packages
with open('requirements.txt') as f:
    requirements = f.read().splitlines()


NAME = 'room_layout_assessment'
AUTHOR = 'DigitalBridge-AI'
VERSION = '0.1'

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    packages=find_packages(exclude=["handler"]),
    install_requires=requirements
)
