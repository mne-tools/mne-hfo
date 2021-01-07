# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CODESPELL_SKIPS ?= "doc/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,install_mne_c.rst,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg"
CODESPELL_DIRS ?= mne_hfo/ doc/ examples/ tests/

all: clean inplace test

clean-pyc:
	find . -name "*.pyc" | xargs rm -f

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build

clean-ctags:
	rm -f tags

clean-cache:
	find . -name "__pycache__" | xargs rm -rf

clean-e:
	find . -name "*-e" | xargs rm -rf

clean: clean-build clean-pyc clean-so clean-ctags clean-cache clean-e

inplace:
	$(PYTHON) setup.py develop

test: inplace check-manifest
	rm -f .coverage
	$(PYTESTS) mne_hfo

test-doc:
	$(PYTESTS) --doctest-modules --doctest-ignore-import-errors mne_hfo

test-coverage:
	rm -rf coverage .coverage
	$(PYTESTS) --cov=mne_hfo --cov-report html:coverage

trailing-spaces:
	find . -name "*.py" | xargs perl -pi -e 's/[ \t]*$$//'

upload-pipy:
	python setup.py sdist bdist_egg register upload

check-manifest:
	check-manifest --ignore .circleci/*,doc,.DS_Store

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_hfo examples tests; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@echo "Running code-spell check"
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=ignore_words.txt $(CODESPELL_DIRS)

type-check:
	mypy ./mne_hfo

pep:
	@$(MAKE) -k flake pydocstyle check-manifest codespell-error type-check

build-doc:
	cd doc; make clean
	cd doc; make html
