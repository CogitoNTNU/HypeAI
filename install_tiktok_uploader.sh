#!/usr/bin/env bash

cd video_automator/video_uploader/tiktok-uploader
hatch build
./venv/bin/python -m pip install -e .
cd ../../..
