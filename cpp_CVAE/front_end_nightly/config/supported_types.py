# supported_types.py

# Argument order matters for constructor generation
SUPPORTED_OBJECTS = {
    "Adam": ["beta1", "beta2", "epsilon"],
}

SUPPORTED_LAYERS = {
    "Linear": ["input_dim", "output_dim", "init_fn"],
    "RELU": [],
}

SUPPORTED_INITIALIZERS = {
    "glorot": "glorot_init",
    "he": "he_init",
    "zeros": "zeros_intt"
}

SUPPORTED_ACTIVATIONS = {
    "relu"
}

SUPPORTED_METADATA_FIELDS = {
    "cell_type", "assay", "development_stage"
}

# Optional: Roll up into a global registry if needed
SUPPORTED = {
    "objects": SUPPORTED_OBJECTS,
    "layers": SUPPORTED_LAYERS,
    "inits": SUPPORTED_INITIALIZERS,
    "activations": SUPPORTED_ACTIVATIONS,
    "metadata_fields": SUPPORTED_METADATA_FIELDS
}
