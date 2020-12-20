# Contributions

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug
free and properly documented (both API docs and an example script),
it can be incorporated into the master branch.

To help developing `mne-hfo`, you will need a few adjustments to your
installation as shown below.

## Running tests

### (Optional) Install development version of MNE-Python
If you want to run the tests with a development version of MNE-Python,
you can install it by running

    $ pip install -U https://github.com/mne-tools/mne-python/archive/master.zip

### Install development version of MNE-HFO 
First, you should [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mne-bids` repository. Then, clone the fork and install it in
"editable" mode.

    $ git clone https://github.com/<your-GitHub-username>/mne-hfo
    $ pip install -e ./mne-hfo

### Install Python packages required to run tests
Install the following packages for testing purposes, plus all optonal MNE-HFO
dependencies to ensure you will be able to run all tests.

    $ pipenv install --dev

### Invoke pytest
Now you can finally run the tests by running `pytest` in the
`mne-bids` directory.

    $ cd mne-bids
    $ pytest

If you have installed the `bids-validator`
on a per-user basis, set the environment variable `VALIDATOR_EXECUTABLE` to point to the path of the `bids-validator` before invoking `pytest`:

    $ VALIDATOR_EXECUTABLE=../bids-validator/bids-validator/bin/bids-validator pytest

## Building the documentation

The documentation can be built using sphinx. For that, please additionally
install the following:

    $ pip install matplotlib nilearn sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot
    
if you don't want to run the examples to build the documentation. This will result in a faster build but produce no plots in the examples.
