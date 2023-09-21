# simple makefile to simplify repetitive build env management tasks under posix

# caution: testing won't work on windows, see README

PYTHON ?= python
PYTESTS ?= pytest
CODESPELL_SKIPS ?= "doc/auto_*,*.fif,*.eve,*.gz,*.tgz,*.zip,*.mat,*.stc,*.label,*.w,*.bz2,*.annot,*.sulc,*.log,*.local-copy,*.orig_avg,*.inflated_avg,*.gii,*.pyc,*.doctree,*.pickle,*.inv,*.png,*.edf,*.touch,*.thickness,*.nofix,*.volume,*.defect_borders,*.mgh,lh.*,rh.*,COR-*,FreeSurferColorLUT.txt,*.examples,.xdebug_mris_calc,bad.segments,BadChannels,*.hist,empty_file,*.orig,*.js,*.map,*.ipynb,searchindex.dat,install_mne_c.rst,plot_*.rst,*.rst.txt,c_EULA.rst*,*.html,gdf_encodes.txt,*.svg,doc/_build*"
CODESPELL_DIRS ?= mne_hfo/ doc/ examples/

all: clean inplace test

clean-so:
	find . -name "*.so" | xargs rm -f
	find . -name "*.pyd" | xargs rm -f

clean-build:
	rm -rf _build
	rm -rf _build
	rm -rf dist
	rm -rf mne_hfo.egg-info

clean-ctags:
	rm -f tags

clean: clean-build clean-so clean-ctags

inplace:
	$(PYTHON) setup.py develop

test: inplace
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

reqs:
	pipfile2req --dev > test_requirements.txt
	pipfile2req > requirements.txt
	pipfile2req > doc/requirements.txt
	pipfile2req --dev >> doc/requirements.txt

flake:
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8"; \
		flake8 --count mne_hfo examples; \
	else \
		echo "flake8 not found, please install it!"; \
		exit 1; \
	fi;
	@echo "flake8 passed"

pydocstyle:
	@echo "Running pydocstyle"
	@pydocstyle

codespell:  # running manually
	@codespell -w -i 3 -q 3 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

codespell-error:  # running on travis
	@echo "Running code-spell check"
	@codespell -i 0 -q 7 -S $(CODESPELL_SKIPS) --ignore-words=.codespellignore $(CODESPELL_DIRS)

isort:
	@if command -v isort > /dev/null; then \
		echo "Running isort"; \
		isort mne_hfo examples doc; \
	else \
		echo "isort not found, please install it!"; \
		exit 1; \
	fi;
	@echo "isort passed"

type-check:
	mypy ./mne_hfo

run-checks:
	isort --check .
	black --check mne_hfo examples
	flake8 .
	mypy ./mne_hfo
	@$(MAKE) pydocstyle
	@$(MAKE) codespell-error
	bibclean-check ./doc/references.bib

build-doc:
	cd doc; make clean
	cd doc; make html
	cd doc; make show

build-pipy:
	python setup.py sdist bdist_wheel

test-pipy:
	twine check dist/*
	twine upload --repository testpypi dist/*
