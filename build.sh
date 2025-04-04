#! /bin/bash
set -e

: ${BUILD_PATH:=build}

python3.10 -m pip install ruff
# remove --exit-zero once all errors are fixed/explicitly ignore
python3.10 -m ruff check --line-length=120 --ignore=F401,E203
# exit when asked to run `ruff` only
if [[ "$1" == "ruff" ]]
then
  exit 0
fi

# Run static code analysis
python3.10 -m pip install mypy
python3.10 -m mypy --no-incremental || true
# exit when asked to run `mypy` only
if [[ "$1" == "mypy" ]]
then
  exit 0
fi

python3.10 setup.py bdist_wheel --dist-dir ${BUILD_PATH}/pip/public/neuronx-distributed-inference