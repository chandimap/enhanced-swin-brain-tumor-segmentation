#!/usr/bin/env bash
CONFIG=$1
mkdir -p runs/{checkpoints,logs}
python src/train.py ${CONFIG}
