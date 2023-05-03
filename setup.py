import setuptools
from setuptools.command.install import install
from setuptools.command.develop import develop
import subprocess
from setuptools.extension import Extension



with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mkidreadoutanalysis",
    version="0.1",
    author="MazinLab, J. Smith et al.",
    author_email="mazinlab@ucsb.edu",
    description="An MKID simulator package for RFSoC",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MKIDReadoutAnalysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"],
)