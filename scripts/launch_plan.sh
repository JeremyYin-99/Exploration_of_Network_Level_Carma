#!/bin/zsh
# Start scripts/run_plan.sh in the background, detached from the terminal and from any Claude
# Code session, with the Mac kept awake (caffeinate). Same arguments as run_plan.sh.
#   WORKERS=8 TAG=price300_grid5 scripts/launch_plan.sh price300 300 --networks grid5
# Progress: tail -f results/params/<plan>/logs/<tag>.log
# Stop:     kill -- -$(cat results/params/<plan>/logs/<tag>.pid)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd $ROOT
PLAN=$1; TAG=${TAG:-$PLAN}
mkdir -p results/params/$PLAN/logs
if pgrep -f "parameter_study.py --plan $PLAN " > /dev/null; then
  echo "Plan $PLAN is already running here."; exit 1
fi
TAG=$TAG WORKERS=${WORKERS:-16} .venv/bin/python -c "
import subprocess, sys
log = 'results/params/$PLAN/logs/$TAG'
p = subprocess.Popen(['caffeinate', '-ims', '/bin/zsh', 'scripts/run_plan.sh'] + sys.argv[1:],
                     start_new_session=True, stdin=subprocess.DEVNULL,
                     stdout=open(log + '.out', 'a'), stderr=subprocess.STDOUT)
open(log + '.pid', 'w').write(str(p.pid))
print(f'Started plan $PLAN ($TAG) with {__import__(\"os\").environ[\"WORKERS\"]} workers, pid {p.pid}.')
" "$@"
