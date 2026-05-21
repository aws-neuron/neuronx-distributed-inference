# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Importing this package applies an upstream bug fix for
# HuggingFaceGenerationAdapter.prepare_inputs_for_generation so that
# adapter.generate() does not raise NameError when forwarding
# tensor_capture_hook downstream. The fix is idempotent and only activates
# if the upstream file still contains the bug.

from . import _upstream_compat  # noqa: F401  (side-effect import)
