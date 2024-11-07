#!/usr/bin/env bash

cd tiktok-uploader
hatch build
python3 -m pip install -e . --break-system-packages
cd ..
