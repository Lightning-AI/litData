.PHONY: test clean docs install-pre-commit install-dependencies setup

# to imitate SLURM set only single node
export SLURM_LOCALID=0
# assume you have installed need packages
export SPHINX_MOCK_REQUIREMENTS=0

setup: install-dependencies install-pre-commit
	@echo "==================== Setup Finished ===================="
	@echo "All set! Ready to go!"

test: clean
	uv pip install -q -r requirements.txt
	uv pip install -q -r requirements/test.txt

	# use this to run tests
	python -m coverage run --source litdata -m pytest src -v --flake8
	python -m coverage report

docs: clean
	uv pip install . --quiet -r requirements/docs.txt
	python -m sphinx -b html -W --keep-going docs/source docs/build

clean:
	# clean all temp runs
	rm -rf $(shell find . -name "mlruns")
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf ./docs/build
	rm -rf ./docs/source/**/generated
	rm -rf ./docs/source/api
	rm -rf ./src/*.egg-info
	rm -rf ./build
	rm -rf ./dist

install-dependencies:
	uv pip install -r requirements.txt
	uv pip install -r requirements/test.txt
	uv pip install -r requirements/docs.txt
	uv pip install -r requirements/extras.txt
	uv pip install -e .


install-pre-commit:
	uv pip install pre-commit
	pre-commit install
