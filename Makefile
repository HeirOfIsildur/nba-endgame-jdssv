# Reproduce every number and figure from a clean clone.
#   make env     create .venv from requirements-lock.txt (needs uv)
#   make all     download -> build -> possessions -> estimate -> maps -> observed -> boot -> figures -> headline
#   make check   Mathematica exact re-solve + break-even identity (needs wolframscript)
PY ?= .venv/bin/python
B ?= 1000
W ?= 4

# stages depend on each other's outputs: never run them in parallel
.NOTPARALLEL:
.PHONY: env all data build possessions estimate maps observed boot figures headline check test paper supplement sensitivity palette
all: data build possessions estimate maps observed boot sensitivity palette figures headline

env:
	uv venv -p 3.13 .venv && uv pip install -p .venv -r requirements-lock.txt && uv pip install -p .venv -e . pytest

data:
	$(PY) -m nbaeg.download 2000 2025
build:
	$(PY) -m nbaeg.build 2000 2025
possessions:
	$(PY) -m nbaeg.possessions
estimate:
	$(PY) -m nbaeg.estimate
maps:
	$(PY) -m nbaeg.bootstrap point
observed:
	$(PY) -m nbaeg.observed
boot:
	$(PY) -m nbaeg.bootstrap run $(B) $(W) 0
	$(PY) -m nbaeg.bootstrap summary $(B)
figures:
	$(PY) -m nbaeg.figures
headline:
	$(PY) -m nbaeg.headline
check:
	$(PY) -m nbaeg.checks
	wolframscript -file checks/dp_check.wl
test:
	$(PY) -m pytest -q tests
paper:
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex
supplement:
	mkdir -p paper/submission
	git archive --format=zip --prefix=nba-endgame-jdssv/ -o paper/submission/nba-endgame-jdssv-code.zip HEAD
sensitivity:
	$(PY) -m nbaeg.sensitivity
	$(PY) -m nbaeg.linkage
palette:
	$(PY) scripts/check_palette.py
