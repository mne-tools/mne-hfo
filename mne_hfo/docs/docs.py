# -*- coding: utf-8 -*-
"""The documentation functions."""
# Authors: Adam Li <adam2392@gmail.com>
#
# License: BSD (3-clause)

from mne.externals.doccer import indentcount_lines

##############################################################################
# Define our standard documentation entries

docdict = dict()

# Detector variables

docdict['sfreq'] = """
sfreq : int | None
    The sampling rate of the data. If ``None``, then the data
    passed to ``fit`` function must be a `mne.io.Raw` object.
"""

docdict['scoring_func'] = """
scoring_func : str
    The scoring function to apply when trying to match HFOs with
    a different dataset, such as manual annotations.
"""

docdict['hfo_name'] = """
hfo_name : str
    What to name the events detected (i.e. fast ripple if freq_band is
    (250, 500)).
"""

# Verbose
docdict['verbose'] = """
verbose : bool, str, int, or None
    If not None, override default verbose level (see :func:`mne.verbose`
    for more info). If used, it should be passed as a
    keyword-argument only."""

# Parallelization
docdict['n_jobs'] = """
n_jobs : int
    The number of jobs to run in parallel (default 1).
    Requires the joblib package.
"""

# Random state
docdict['random_state'] = """
random_state : None | int | instance of ~numpy.random.RandomState
    If ``random_state`` is an :class:`int`, it will be used as a seed for
    :class:`~numpy.random.RandomState`. If ``None``, the seed will be
    obtained from the operating system (see
    :class:`~numpy.random.RandomState` for details). Default is
    ``None``.
"""

docdict_indented = {}


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated ``__doc__``.
    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    if len(lines) < 2:
        icount = 0
    else:
        icount = indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = ' ' * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]]
                for line in lines[1:]:
                    newlines.append(indent + line)
                indented[name] = '\n'.join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split('\n')[0] if funcname is None else funcname
        raise RuntimeError('Error documenting %s:\n%s'
                           % (funcname, str(exp)))
    return f
