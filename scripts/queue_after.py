"""Start a plan once another plan has finished on this Mac, detached and kept awake.

    .venv/bin/python scripts/queue_after.py <plan to wait for> <workers> <tag> <plan> <days> [args]
"""
import os
import subprocess
import sys

wait_for, workers, tag, plan, days, *rest = sys.argv[1:]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# "[.]" keeps pgrep from matching this waiter's own command line.
pattern = f"parameter_study[.]py --plan {wait_for} "
cmd = (f'while pgrep -f "{pattern}" > /dev/null; do sleep 30; done; '
       f'WORKERS={workers} TAG={tag} zsh scripts/launch_plan.sh {plan} {days} {" ".join(rest)}')
os.makedirs(os.path.join(root, "results", "params"), exist_ok=True)
log = open(os.path.join(root, "results", "params", f"queue_{tag}.log"), "a")
p = subprocess.Popen(["caffeinate", "-ims", "/bin/zsh", "-c", cmd], cwd=root, start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
print(f"queued {plan} ({tag}) after {wait_for}: waiter pid {p.pid}")
