#! /bin/bash
set -e

: ${BUILD_PATH:=build}

python -m pip install ruff
# remove --exit-zero once all errors are fixed/explicitly ignore
python -m ruff check --line-length=120 --ignore=F401,E203
# exit when asked to run `ruff` only
if [[ "$1" == "ruff" ]]
then
  exit 0
fi

# Run static code analysis
python -m pip install mypy
python -m mypy --no-incremental || true
# exit when asked to run `mypy` only
if [[ "$1" == "mypy" ]]
then
  exit 0
fi

python setup.py bdist_wheel --dist-dir ${BUILD_PATH}/pip/public/neuronx-distributed-inference