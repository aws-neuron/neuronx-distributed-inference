import os
import pytest
import torch

@pytest.fixture(autouse=True)
def set_constant_seed():
    """
    Sets a constant seed before each test runs to ensure deterministic behavior between runs.

    Without a constant seed, the random weights differ between runs, which can result in logit
    validation failing due to varying precision loss accumulation from one run to the next.
    Therefore, a constant seed improves test stability.
    """
    torch.manual_seed(0)


@pytest.fixture(autouse=True)
def set_unique_compiler_workdir(request, monkeypatch):
    base_compile_workdir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")
    unique_compile_workdir = os.path.join(base_compile_workdir, get_test_path_name(request)) + "/"
    monkeypatch.setenv("BASE_COMPILE_WORK_DIR", unique_compile_workdir)
    print(f"Using compiler work dir: {unique_compile_workdir}")


def get_test_path_name(request):
    components = request.node.nodeid.split("::")
    test_file_name = os.path.splitext(os.path.basename(components[0]))[0]
    valid_file_name_parts = [get_valid_file_name(component) for component in components[1:]]
    return "_".join([test_file_name] + valid_file_name_parts)


def get_valid_file_name(input):
    return "".join([c if c.isalnum() else "_" for c in input])
