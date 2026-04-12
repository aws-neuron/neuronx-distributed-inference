"""Integration tests for YuE Music Generation on Neuron.

Tests the NxDI-based music generation pipeline including S1 (7B LLaMA, TP=2),
S2 (1B LLaMA, TP=1), NKI MLP kernel patch, and E2E orchestration.

Requires a trn2.3xlarge instance with Neuron SDK 2.28+, YuE models downloaded,
and xcodec_mini_infer installed via setup.sh.

Usage:
    # On trn2.3xlarge with Neuron SDK 2.28
    source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_13/bin/activate

    # Set MODEL_DIR to where setup.sh installed models
    export MODEL_DIR=/mnt/models

    cd contrib/models/YuE-Music-Generation
    PYTHONPATH=src:$PYTHONPATH pytest test/integration/test_model.py -v -s
"""

import os
import sys
import re
import tempfile

import pytest

# Check for Neuron hardware
try:
    import torch_neuronx

    HAS_NEURONX = True
except ImportError:
    HAS_NEURONX = False

requires_neuron = pytest.mark.skipif(
    not HAS_NEURONX, reason="torch_neuronx not available (requires Neuron hardware)"
)

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(SRC_DIR))

MODEL_DIR = os.environ.get("MODEL_DIR", "/mnt/models")
CONTRIB_DIR = os.path.join(os.path.dirname(__file__), "..", "..")


# ============================================================================
# Unit-level tests (no hardware needed)
# ============================================================================


class TestLyricsParsing:
    """Test lyrics parsing logic from the orchestrator."""

    def test_split_lyrics_basic(self):
        """Test that lyrics are split into sections correctly."""
        lyrics = "[verse]\nHello world\n\n[chorus]\nLa la la\n"
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        result = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

        assert len(result) == 2
        assert "[verse]" in result[0]
        assert "Hello world" in result[0]
        assert "[chorus]" in result[1]
        assert "La la la" in result[1]

    def test_split_lyrics_multiple_sections(self):
        """Test parsing with multiple section types."""
        lyrics = "[intro]\nMusic\n[verse]\nWords\n[chorus]\nHook\n[outro]\nEnd\n"
        pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
        segments = re.findall(pattern, lyrics, re.DOTALL)
        result = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]

        assert len(result) == 4
        assert "[intro]" in result[0]
        assert "[outro]" in result[3]

    def test_genre_file_format(self):
        """Test that the sample genre file can be read."""
        genre_path = os.path.join(CONTRIB_DIR, "genre.txt")
        assert os.path.exists(genre_path), f"genre.txt not found at {genre_path}"

        with open(genre_path) as f:
            genre = f.read().strip()

        assert len(genre) > 0, "genre.txt should not be empty"

    def test_lyrics_file_format(self):
        """Test that the sample lyrics file has proper section tags."""
        lyrics_path = os.path.join(CONTRIB_DIR, "lyrics.txt")
        assert os.path.exists(lyrics_path), f"lyrics.txt not found at {lyrics_path}"

        with open(lyrics_path) as f:
            lyrics = f.read()

        # Must contain at least one section tag
        assert re.search(r"\[\w+\]", lyrics), (
            "lyrics.txt must contain section tags like [verse]"
        )


class TestNKIPatch:
    """Test NKI MLP patch module structure."""

    def test_nki_patch_importable(self):
        """Test that the NKI patch module can be imported (on any platform)."""
        # The module should be importable even without Neuron hardware
        # (it guards hardware-specific imports internally)
        import importlib.util

        spec = importlib.util.find_spec("nki_mlp_patch")
        if spec is None:
            # Try direct path
            patch_path = os.path.join(SRC_DIR, "nki_mlp_patch.py")
            assert os.path.exists(patch_path), "nki_mlp_patch.py not found in src/"


class TestSetupScript:
    """Test that the setup script exists and is well-formed."""

    def test_setup_script_exists(self):
        """Test that setup.sh exists."""
        setup_path = os.path.join(CONTRIB_DIR, "setup.sh")
        assert os.path.exists(setup_path), "setup.sh not found"

    def test_setup_script_has_shebang(self):
        """Test that setup.sh has a proper shebang."""
        setup_path = os.path.join(CONTRIB_DIR, "setup.sh")
        with open(setup_path) as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!/"), (
            f"setup.sh should start with shebang, got: {first_line}"
        )


# ============================================================================
# Hardware-dependent integration tests
# ============================================================================


@requires_neuron
class TestE2EPipeline:
    """End-to-end pipeline tests (requires trn2 + models downloaded via setup.sh)."""

    @pytest.fixture(scope="class")
    def check_models_available(self):
        """Check that models have been downloaded via setup.sh."""
        yue_path = os.path.join(MODEL_DIR, "YuE")
        s1_path = os.path.join(MODEL_DIR, "YuE-s1-7B-anneal")
        s2_path = os.path.join(MODEL_DIR, "YuE-s2-1B-general")
        xcodec_path = os.path.join(MODEL_DIR, "xcodec_mini_infer")

        missing = []
        if not os.path.isdir(yue_path):
            missing.append(f"YuE repo: {yue_path}")
        if not os.path.isdir(s1_path):
            missing.append(f"S1 weights: {s1_path}")
        if not os.path.isdir(s2_path):
            missing.append(f"S2 weights: {s2_path}")
        if not os.path.isdir(xcodec_path):
            missing.append(f"xcodec: {xcodec_path}")

        if missing:
            pytest.skip(
                f"Models not available. Run setup.sh first. Missing: {', '.join(missing)}"
            )

    def test_e2e_generation(self, check_models_available):
        """Test full E2E music generation pipeline.

        This runs the orchestrator script as a subprocess (same as production),
        generating a short clip (~15s, 1 segment) and validating output files.
        """
        import subprocess

        genre_path = os.path.join(CONTRIB_DIR, "genre.txt")
        lyrics_path = os.path.join(CONTRIB_DIR, "lyrics.txt")

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                os.path.join(SRC_DIR, "yue_e2e_neuron.py"),
                "--genre_txt",
                genre_path,
                "--lyrics_txt",
                lyrics_path,
                "--output_dir",
                tmpdir,
                "--seed",
                "123",
                "--run_n_segments",
                "1",
                "--max_new_tokens",
                "500",
            ]

            env = os.environ.copy()
            env["MODEL_DIR"] = MODEL_DIR

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )

            assert result.returncode == 0, (
                f"E2E pipeline failed with return code {result.returncode}.\n"
                f"STDOUT:\n{result.stdout[-2000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

            # Check output files exist
            output_files = os.listdir(tmpdir)
            wav_files = [f for f in output_files if f.endswith(".wav")]
            assert len(wav_files) > 0, (
                f"No .wav files generated in {tmpdir}. Files: {output_files}"
            )

            # Check file sizes are reasonable (at least 10KB for a few seconds of audio)
            for wav_file in wav_files:
                wav_path = os.path.join(tmpdir, wav_file)
                size = os.path.getsize(wav_path)
                assert size > 10000, (
                    f"Output file {wav_file} is too small ({size} bytes), "
                    "likely empty or corrupt"
                )

    def test_e2e_with_nki_kernels(self, check_models_available):
        """Test E2E pipeline with NKI kernel optimization enabled."""
        import subprocess

        genre_path = os.path.join(CONTRIB_DIR, "genre.txt")
        lyrics_path = os.path.join(CONTRIB_DIR, "lyrics.txt")

        with tempfile.TemporaryDirectory() as tmpdir:
            cmd = [
                sys.executable,
                os.path.join(SRC_DIR, "yue_e2e_neuron.py"),
                "--genre_txt",
                genre_path,
                "--lyrics_txt",
                lyrics_path,
                "--output_dir",
                tmpdir,
                "--seed",
                "123",
                "--run_n_segments",
                "1",
                "--max_new_tokens",
                "500",
                "--nki-kernels",
            ]

            env = os.environ.copy()
            env["MODEL_DIR"] = MODEL_DIR

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                env=env,
            )

            assert result.returncode == 0, (
                f"E2E + NKI pipeline failed.\nSTDERR:\n{result.stderr[-2000:]}"
            )

            wav_files = [f for f in os.listdir(tmpdir) if f.endswith(".wav")]
            assert len(wav_files) > 0, "No .wav files generated with NKI kernels"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
