#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/ssd/ssd_googlenet_solver.prototxt $@
