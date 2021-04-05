# Contributions

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug free and properly documented (both API docs
and an example script), it can be incorporated into the master branch.

To help developing `mne-hfo`, you will need a few adjustments to your installation as shown below.
We heavily rely on ``mne``, so one can even peruse their [contributing guide](https://mne.tools/stable/install/contributing.html#:~:text=MNE%2DPython%20is%20maintained%20by,(even%20just%20typo%20corrections)).

## Basic Setup

The basic setup for development will be: i) [Git](https://github.com/git-guides/install-git), which is 
a command-line tool that enables version control, and ii) a Python virtual environment.

For a Python virtual environment, one can use [miniconda](https://docs.conda.io/en/latest/miniconda.html),
or [pip](https://pip.pypa.io/en/stable/installing/). We recommend ``miniconda`` for Mac/Windows users.

Most of these tools will use the command line. For Windows 10 users, there exists a linux sub-system https://docs.microsoft.com/en-us/windows/wsl/install-win10

### Forking Repository

Once you have git installed, you should create a [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) of the ``mne-hfo``
repository. This will setup a "copy" of the current state of the repository to allow you to 
freely push somewhere. You are unable to directly push to the ``mne-tools/mne-hfo`` repository for 
obvious security reasons.

### Clone Repository

Next, you should clone your repository locally and add the upstream copy of ``mne-hfo`` to your 
list of git "remotes". Afterwards, you can see where git is 
pulling from by typing in ``git remote -v``. For example, my local copy will say:

    (base) adam2392@Adams-MacBook-Pro mne-hfo % git remote -v
    origin	https://github.com/adam2392/mne-hfo.git (fetch)
    origin	https://github.com/adam2392/mne-hfo.git (push)

Meaning that my remote version is called ``origin`` and the url that it is 
fetching and pushing to is `https://github.com/adam2392/mne-hfo.git`. Since you are 
on a fork, you should also add the ``upstream`` version:

    # add upstream remote
    git remote add upstream https://github.com/mne-tools/mne-hfo.git

### Installation

Now that you have your repository settings setup, you should install the development version 
of ``mne-hfo``. You should first create a virtual environment for Python3.6+ say using conda:

    # create a conda environment called "mne-hfo"
    # optionally add the "python=3.8" flag to tell it which python version to use.
    conda create -n mne-hfo

    # activate the said virtual environment
    conda activate mne-hhfo

    # install libraries
    pip install -r requirements.txt

    # install development libraries
    pip install -r test_requirements.txt

Alternatively, one could use ``pipenv`` to install libraries. One should only use this method if 
they're familiar with ``pip`` and ``pipenv``. First, cd to the root of the cloned repo:

    cd <path_to_mne_hfo_dir>

Then install packages using pipenv:

    python3.8 -m venv .venv

    pip install --upgrade pip pipenv

    pipenv install --skip-lock
    pipenv install --dev --skip-lock

## Running tests

### (Optional) Install development version of MNE-Python

If you want to run the tests with a development version of MNE-Python, you can install it by running

    pip install -U https://github.com/mne-tools/mne-python/archive/master.zip

### Invoke pytest

Now you can finally run the tests by running `pytest` in the
`mne-hfo` directory.
    
    # run pytests
    cd mne-hfo
    pytest ./tests

## Building the documentation

The documentation can be built using sphinx. For that, please additionally install the following:

    pip install matplotlib nilearn sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow

To build the documentation locally, one can run:

    cd doc/
    make html

or

    make html-noplot

if you don't want to run the examples to build the documentation. This will result in a faster build but produce no
plots in the examples.

### Building tutorials

Our tutorials will be written as jupyter ipython notebooks. To make your python virtual environment available,
install ``ipykernel`` and run:

    python -m ipykernel install --name mnehfo --user 

For some of the tutorials, we use a test dataset that is already in BIDS format. The ground-truth is from publication (
reference in README).

# Notes on Detector Design

The amount of HFO code and algorithms out there is quite staggering. In order to make an easy to use package, with a
standard API, we conform each of the HFO ``Detector`` to
be [scikit-learn compatible](https://scikit-learn.org/stable/developers/develop.html), which can then leverage the
entire ``scikit-learn`` API, such as ``Pipeline``, ``GridSearchCV``, and more.

In addition, we develop our Detectors to work
with [mne-python Raw objects](https://mne.tools/stable/generated/mne.io.Raw.html)
which are just very robust data structure for EEG data.

Finally, we assume that the datasets we work with are
generally [BIDS-compliant](https://bids-specification.readthedocs.io/en/stable/). We structure all HFO output as a
task ``*events.tsv`` file, which stores the HFO events according to the events
directive (https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/05-task-events.html).
