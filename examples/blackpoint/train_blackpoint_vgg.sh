#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=examples/blackpoint/blackpoint_vgg_solver.prototxt $@
