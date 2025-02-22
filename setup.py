from setuptools import find_packages,setup
from typing import List


HYPHEN_E_DOT = "-e ."

def get_requirements(file_path:str)->List[str]:
    """this function returns list of requirements from the file"""
    with open(file_path) as f:
        requirements = f.read().splitlines()
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author='kalyan',
    author_email='bskalyansrinivas@gmail.com',
    packages=find_packages(),
    include_requires=get_requirements('requirements.txt'),
)
