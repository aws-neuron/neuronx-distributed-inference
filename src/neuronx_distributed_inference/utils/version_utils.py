import packaging
import torch_neuronx


def get_torch_neuronx_build_version() -> packaging.version.Version:
    """
    Retrieves the currently installed torch_neuronx build version.
    This build version is useful for checking torch_neuronx feature availability.

    torch-neuronx versions follow this format: <torch-xla-version>.<build-version>
    For example, version 2.7.0.2.11.14773+100ca7de is compatible with torch-xla 2.7.0
    and its build version is 2.11.14773+100ca7de.
    """
    version = torch_neuronx.__version__
    version_parts = version.split(".", 3)
    assert len(version_parts) >= 4, f"Unexpected torch_neuronx version: {version}"

    build_version = packaging.version.parse(version_parts[3])
    return build_version
