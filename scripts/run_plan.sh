#!/bin/zsh
# Run one plan of controller/parameter_study.py in the foreground, logging to
# results/params/<plan>/logs/. Resumable: rerunning skips every finished day.
#   WORKERS=16 scripts/run_plan.sh <plan> <learning days> [more parameter_study arguments]
#   e.g. scripts/run_plan.sh price300 300 --networks grid5
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
export SUMO_HOME=$ROOT/sumo
PLAN=$1; DAYS=$2; shift 2
TAG=${TAG:-$PLAN}
LOG=$ROOT/results/params/$PLAN/logs
mkdir -p $LOG
WORKERS=${WORKERS:-16}
echo "START $(date '+%Y-%m-%d %H:%M:%S') $(hostname -s) workers $WORKERS args $*" >> $LOG/phases_$TAG.log
.venv/bin/python controller/parameter_study.py --plan $PLAN --days $DAYS --workers $WORKERS "$@" >> $LOG/$TAG.log 2>&1
rc=$?
echo "DONE $(date '+%Y-%m-%d %H:%M:%S') $(hostname -s) exit $rc" >> $LOG/phases_$TAG.log
