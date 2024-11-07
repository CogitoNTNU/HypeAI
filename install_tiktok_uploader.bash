#!/usr/bin/env bash

cd tiktok-uploader
hatch build
./../venv/bin/python -m pip install -e .
cd ..
