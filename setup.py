from setuptools import PEP420PackageFinder, setup
import os
import subprocess
from subprocess import CalledProcessError


def get_version(version_str):
    major, minor, patch = version_str.split(".")
    patch = os.getenv('VERSION_PATCH', patch)
    suffix = os.getenv('SUFFIX')
    if not suffix:
        try:
            suffix = f'{subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()[0:8]}.dev'
        except CalledProcessError:
            suffix = 'dev'
    return f"{major}.{minor}.{patch}+{suffix}"


exec(open("src/neuronx_distributed_inference/_version.py").read())
setup(
    name="neuronx-distributed-inference",
    version=get_version(__version__),  # noqa F821
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="aws neuron",
    packages=PEP420PackageFinder.find(where="src"),
    package_data={"": []},
    install_requires=[
        "neuronx_distributed",
        "transformers==4.51.*",
        "huggingface-hub",
        "sentencepiece",
        "torchvision",
        "pillow",
        "blobfile",
    ],
    extras_require={
        "test": ["pytest", "pytest-forked", "pytest-cov", "pytest-xdist", "accelerate"],
    },
    python_requires=">=3.7",
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "inference_demo=neuronx_distributed_inference.inference_demo:main",
            "nxdi_distributed_launcher=neuronx_distributed_inference.scripts.nxdi_distributed_launcher:main",
        ],
    },
)
