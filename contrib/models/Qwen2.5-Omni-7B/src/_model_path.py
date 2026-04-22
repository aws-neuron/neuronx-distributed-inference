# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Helper for resolving the Qwen2.5-Omni-7B weight path.
#
# Honors ``$QWEN25_OMNI_MODEL_PATH`` if it points at a directory with a
# ``config.json``. Otherwise delegates to ``huggingface_hub.snapshot_download``
# which is a no-op if the model is already cached and returns the real snapshot
# directory (including the commit hash) in either case.

import os


HF_REPO_ID = "Qwen/Qwen2.5-Omni-7B"


def resolve_model_path() -> str:
    env = os.environ.get("QWEN25_OMNI_MODEL_PATH")
    if env and os.path.isfile(os.path.join(env, "config.json")):
        return env
    from huggingface_hub import snapshot_download
    return snapshot_download(HF_REPO_ID)
