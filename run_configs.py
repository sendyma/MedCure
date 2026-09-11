# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Named experiment configs.

Each function returns a `Config` and its name is both the value passed to
``--config_name`` and the sub-directory the run is written to.

    python -m main --config_name medcure
"""

from configs import Config


def medcure():
    return Config()
