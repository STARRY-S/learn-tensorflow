#!/bin/bash

cd $(dirname $0)/../

set -exuo pipefail

# Prepare virtual env

python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
