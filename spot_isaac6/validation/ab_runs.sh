#!/bin/bash
# The open-floor A/B runs behind validation/physx_open_floor (legcol60, cmd 0.5 m/s, 10 s). Inside the Isaac container:
#   bash spot_isaac6/validation/ab_runs.sh
# The implicit variant needs its own stage: python spot_isaac6/make_spot_stage.py --open-floor --act implicit spot_isaac6/stages/spot_floor_implicit.usda
W=$(cd "$(dirname "$0")/.." && pwd); O=$W/runs/ab; R="/isaac-sim/python.sh $W/walk.py"
cd /tmp
$R --out $O/explicit                                         2>&1 | grep -E "\[walk\] RESULT|Traceback"
$R --out $O/implicit --act implicit --stage $W/stages/spot_floor_implicit.usda --no-video 2>&1 | grep -E "\[walk\] RESULT|Traceback"
$R --out $O/explicit_lag1 --lag 1 --no-video                 2>&1 | grep -E "\[walk\] RESULT|Traceback"
$R --out $O/explicit_noarmfix --no-arm-fix --no-video        2>&1 | grep -E "\[walk\] RESULT|Traceback"
