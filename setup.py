#! /usr/bin/env python
"""Setup MNE-HFO."""
import os
from setuptools import setup, find_packages

# get the version
version = None
with open(os.path.join('mne_hfo', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')


descr = ('MNE-HFO: Facilitates estimation/detection of high-frequency oscillation'
         'events on iEEG data with MNE-Python and MNE-BIDS.')

DISTNAME = 'mne-hfo'
DESCRIPTION = descr
MAINTAINER = 'Adam Li'
MAINTAINER_EMAIL = 'adam2392@gmail.com'
URL = 'https://mne.tools/mne-hfo/'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/adam2392/mne-hfo.git'
VERSION = version

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          long_description_content_type='text/markdown',
          python_requires='~=3.6',
          install_requires=[
              'mne >=0.22',
              'mne-bids >= 0.6',
              'numpy >=1.14',
              'scipy >=0.18.1',
              'scikit-learn >= 0.23'
          ],
          extras_require={
              'full': [
                  'joblib >= 1.0.0',
                  'matplotlib',
                  'pandas >=0.23.4'
              ]
          },
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
              'Programming Language :: Python :: 3.8',
              'Programming Language :: Python :: 3.9',
          ],
          platforms='any',
          packages=find_packages(),
          entry_points={'console_scripts': [
              'mne_hfo = mne_hfo.commands.run:main',
          ]},
          project_urls={
              'Documentation': URL,
              'Bug Reports': 'https://github.com/mne-tools/mne-hfo/issues',
              'Source': 'https://github.com/mne-tools/mne-hfo',
          },
          )
