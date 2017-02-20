#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/lstm/lstm_bits_solver.prototxt $@
