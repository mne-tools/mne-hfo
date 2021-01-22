# Contributions

Contributions are welcome in the form of pull requests.

Once the implementation of a piece of functionality is considered to be bug free and properly documented (both API docs
and an example script), it can be incorporated into the master branch.

To help developing `mne-hfo`, you will need a few adjustments to your installation as shown below.

## Running tests

### (Optional) Install development version of MNE-Python

If you want to run the tests with a development version of MNE-Python, you can install it by running

    $ pip install -U https://github.com/mne-tools/mne-python/archive/master.zip

### Install development version of MNE-HFO

First, you should [fork](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the `mne-hfo`
repository. Then, clone the fork and install it in
"editable" mode.

    $ git clone https://github.com/<your-GitHub-username>/mne-hfo
    $ pip install -e ./mne-hfo

### Install Python packages required to run tests

Install the following packages for testing purposes, plus all optonal MNE-HFO dependencies to ensure you will be able to
run all tests.

    $ pipenv install --dev

### Invoke pytest

Now you can finally run the tests by running `pytest` in the
`mne-hfo` directory.

    $ cd mne-hfo
    $ pytest

## Building the documentation

The documentation can be built using sphinx. For that, please additionally install the following:

    $ pip install matplotlib nilearn sphinx numpydoc sphinx-gallery sphinx_bootstrap_theme pillow

To build the documentation locally, one can run:

    $ cd doc/
    $ make html

or

    $ make html-noplot

if you don't want to run the examples to build the documentation. This will result in a faster build but produce no
plots in the examples.

### Building tutorials

Our tutorials will be written as jupyter ipython notebooks. To make your python virtual environment available,
install ``ipykernel`` and run:

    python -m ipykernel install --name mnehfo --user 

For some of the tutorials, we use a test dataset that is already in BIDS format. The ground-truth is from publication (
reference in README).

## Notes on Detector Design

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
