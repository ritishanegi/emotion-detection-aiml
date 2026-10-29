#!/usr/bin/env python3
"""
Loop run.py for job_array_task_id 1..15 with fixed args:
  run_version_number=8, task=binary, toy=False, run_modelN=0

Continues on failure, logs errors, prints a final summary.
"""

import subprocess
import sys
import os

PYTHON = r"C:\Users\Sonu\AppData\Local\Python\pythoncore-3.14-64\python.exe"
SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
CWD    = os.path.dirname(os.path.abspath(__file__))

FIXED_ARGS = [
    "--run_version_number=8",
    "--task=binary",
    "--toy=False",
    "--run_modelN=0",
]

TOTAL = 15

# Map job_array_task_id -> subreddit name (run_version_number=8 order)
SUBREDDITS = [
    "addiction", "EDAnonymous", "adhd", "autism", "alcoholism",
    "bipolarreddit", "bpd", "depression", "anxiety", "ptsd",
    "healthanxiety", "lonely", "schizophrenia", "socialanxiety", "suicidewatch",
]

successes = []
failures  = []

for job_id in range(1, TOTAL + 1):
    sr = SUBREDDITS[job_id - 1]
    cmd = [PYTHON, SCRIPT, f"--job_array_task_id={job_id}"] + FIXED_ARGS

    print(f"\n{'='*65}")
    print(f"  [{job_id:02d}/{TOTAL}]  subreddit={sr}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*65}\n", flush=True)

    result = subprocess.run(cmd, cwd=CWD, capture_output=False)

    if result.returncode == 0:
        print(f"\n  [OK]   job {job_id} ({sr}) SUCCEEDED\n", flush=True)
        successes.append((job_id, sr))
    else:
        print(f"\n  [FAIL] job {job_id} ({sr}) FAILED  (exit code {result.returncode})\n", flush=True)
        failures.append((job_id, sr, result.returncode))

# ── Final summary ──────────────────────────────────────────────
print("\n" + "="*65)
print(f"  FINAL SUMMARY:  {len(successes)}/{TOTAL} succeeded,  {len(failures)}/{TOTAL} failed")
print("="*65)
print(f"\n  Succeeded ({len(successes)}):")
for job_id, sr in successes:
    print(f"    [{job_id:02d}] {sr}")

if failures:
    print(f"\n  Failed ({len(failures)}):")
    for job_id, sr, code in failures:
        print(f"    [{job_id:02d}] {sr}  (exit {code})")
else:
    print("\n  All runs completed without errors.")
print()
