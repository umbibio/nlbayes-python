# all .pyx files in a folder
import setuptools
from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext
from Cython.Build import cythonize
# import Cython.Build
# from Cython.Distutils import Extension
from glob import glob



libs = {
    'libraries': ['gsl', 'gslcblas', 'm'],
    'include_dirs': [
        'libnlbayes/include',
    ]
}

args = {
    'extra_compile_args': ['-fopenmp'],
    'extra_link_args': ['-lgomp']
}

all_lib = [ext for ext in glob('libnlbayes/include/*.h')]
all_cpp = [ext for ext in glob('libnlbayes/src/*.cpp')]
all_dep = all_lib + all_cpp

extensions = [ Extension("nlbayes.ModelORNOR", ["nlbayes/ModelORNOR.pyx"]+all_cpp, depends=all_dep, language="c++", **libs, **args)]


with open("README.md", "r") as fh:
    long_description = fh.read()


import re
VERSIONFILE="nlbayes/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))


setup(
    name = 'nlbayes',
    version=verstr,
    author="Argenis Arriojas",
    author_email="arriojasmaldonado001@umb.edu",
    description=(
        'A Bayesian Networks approach for infering active Transcription Factors '
        'using logic models of transcriptional regulation'),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/umbibio/nlbayes',
    packages=setuptools.find_packages(),
    license='MIT',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Cython",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    entry_points = {
        'console_scripts': [
            'nlb-ornor-inference=nlbayes.commands.ornor_inference:main',
            'nlb-simulation=nlbayes.commands.simulation:main',
        ],
    },
    cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(extensions, annotate=False, language_level=3),
    setup_requires=[
        'cython',
        'cysignals',
    ],
    install_requires=[
        'cython',
        'cysignals',
        'numpy',
        'pandas',
        'scipy',
        'psutil',
        'tqdm',
        'CythonGSL@https://github.com/twiecki/CythonGSL/archive/master.zip',
    ],
    zip_safe=False,
)
