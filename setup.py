from setuptools import setup
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, 'README.rst')) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, 'README.rst'), encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='spectralgp',
    version='alpha',
    description=('SpectralGP repo'),
    long_description=long_description,
    author='Greg Benton, Wesley Maddox, Jayson Salkey, Julio Albinati, Andrew Gordon Wilson',
    author_email='wm326@cornell.edu',
    url='https://github.com/wjmaddox/spectralgp',
    license='MPL-2.0',
    packages=['spectralgp'],
   install_requires=[
    'matplotlib==3.0.3',
    'setuptools==41.0.0',
    'scipy>=1.2.1',
    'torch>=1.0.1',
    'numpy==1.16.2',
    'pandas==0.24.2',
    'gpytorch>=0.3.1',
    'rpy2>=3.0.3',
    'scikit_learn>=0.20.3'
   ],
    include_package_data=True,
    classifiers=[
        'Development Status :: 0',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.7'],
)