from setuptools import PEP420PackageFinder, setup

exec(open("src/neuronx_distributed_inference/_version.py").read())
setup(
    name="neuronx-distributed-inference",
    version=__version__,  # noqa F821
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
    install_requires=["neuronx_distributed", "transformers==4.43.*", "sentencepiece"],
    python_requires=">=3.7",
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "inference_demo=neuronx_distributed_inference.inference_demo:main",
        ],
    },
)
