import pytest
import packaging.version
from unittest.mock import patch

from neuronx_distributed_inference.utils.version_utils import get_torch_neuronx_build_version


class TestGetTorchNeuronxBuildVersion:
    """Test suite for get_torch_neuronx_build_version function."""

    @pytest.mark.parametrize(
        "version, build_version",
        [
            ("2.7.0.2.11.14773", "2.11.14773"),
            ("2.7.0.2.11.14773+100ca7de", "2.11.14773+100ca7de"),
        ]
    )
    def test_valid_version(self, version, build_version):
        with patch('neuronx_distributed_inference.utils.version_utils.torch_neuronx') as mock_torch_neuronx:
            mock_torch_neuronx.__version__ = version

            result = get_torch_neuronx_build_version()

            assert isinstance(result, packaging.version.Version)
            assert str(result) == build_version

    @pytest.mark.parametrize(
        "version",
        [
            "2.7.0+100ca7de",
            "2.7.0",
            "2.7",
            "2",
            "",
        ]
    )
    def test_invalid_torch_neuronx_version(self, version):
        with patch('neuronx_distributed_inference.utils.version_utils.torch_neuronx') as mock_torch_neuronx:
            mock_torch_neuronx.__version__ = version

            with pytest.raises(AssertionError) as exc_info:
                get_torch_neuronx_build_version()

            assert f"Unexpected torch_neuronx version: {version}" in str(exc_info.value)

    @pytest.mark.parametrize(
        "version",
        [
            "2.7.0.invalid..version",
            "2.7.0.2.11.14773.extra.parts+100ca7de",
        ]
    )
    def test_invalid_build_version(self, version):
        with patch('neuronx_distributed_inference.utils.version_utils.torch_neuronx') as mock_torch_neuronx:
            mock_torch_neuronx.__version__ = version
            
            # This should raise a packaging.version.InvalidVersion exception
            with pytest.raises(packaging.version.InvalidVersion):
                get_torch_neuronx_build_version()
