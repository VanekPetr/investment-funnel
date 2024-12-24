#!/bin/bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# do not update the uv.lock file
uv sync -vv --frozen
