# S3Diff one-step super-resolution on AWS Neuron
# Uses torch_neuronx.trace() for compilation -- not NxDI TP sharding
# (model is ~2 GB, fits on a single NeuronCore)
