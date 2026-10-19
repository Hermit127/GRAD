#!/usr/bin/env python3
"""GPU watcher: grab an idle GPU the moment it frees up and run Table 5 jobs.

Runs on the GPU host (launched by launch_table5_gpu.sh). One worker per GPU;
each worker claims jobs atomically from jobs_table5.txt and skips finished ones.

Free policy:
  strict (default): memory.used < --mem-max AND utilization < --util-max for
                    --stable consecutive polls (protects others' jobs);
  share (after --share-after seconds without a start): memory free >=
                    --share-min-free and utilization <= --share-util, stable
                    for 3 polls; lets us use spare capacity on a busy card.
Workers are retried/respawned as long as jobs remain and the GPU is free again.

Writes: code/repro/out/table5_ablation/watcher_status.json (every poll) and
watcher.log; exits when every job is done/failed or --max-runtime is exceeded.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(os.environ.get("GRAD_BASE", "."))
WS = os.environ.get("TABLE5_WS", "code/repro/out/table5_ablation")
QUEUE = REPO / WS
JOBS = QUEUE / os.environ.get("TABLE5_JOBS", "jobs_table5.txt")
LOCKS = QUEUE / "locks"
RUNS = QUEUE / os.environ.get("TABLE5_RUNS", "runs")
WORKER = REPO / os.environ.get(
    "TABLE5_WORKER", "code/repro/final_five_tables/run_table5_ablation_gpu.sh")
STATUS = QUEUE / "watcher_status.json"
WLOG = QUEUE / "watcher.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    with open(WLOG, "a") as f:
        f.write(line + "\n")


def query_gpus() -> list[dict]:
    out = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,uuid",
        "--format=csv,noheader,nounits"], text=True)
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util, uuid = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "mem_used": int(used),
                     "mem_total": int(total), "util": int(util), "uuid": uuid})
    # any compute process (ours or others') counts as occupied -> never share by default
    busy_uuids = set()
    try:
        apps = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits"], text=True).strip()
        for line in apps.splitlines():
            if line.strip():
                busy_uuids.add(line.split(",")[0].strip())
    except Exception:
        pass
    for g in gpus:
        g["has_compute_app"] = g["uuid"] in busy_uuids
    return gpus


def load_jobs() -> list[str]:
    jobs = []
    for line in JOBS.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4:      # replication: dataset cell variant seed
            jobs.append("_".join([parts[0], parts[1], parts[2], f"s{parts[3]}"]))
        elif len(parts) == 3:    # phase2: candidate_label fold seed
            jobs.append("_".join([parts[0], parts[1], f"s{parts[2]}"]))
        elif len(parts) == 2:    # panelb: order seed
            jobs.append(f"{parts[0]}_s{parts[1]}")
    return jobs


def remaining(jobs: list[str]) -> list[str]:
    out = []
    for key in jobs:
        if (RUNS / f"{key}.json").exists():
            continue
        if (LOCKS / f"{key}.done").exists() or (LOCKS / f"{key}.failed").exists():
            continue
        out.append(key)
    return out


def main() -> int:
    QUEUE.mkdir(parents=True, exist_ok=True)
    LOCKS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    if not JOBS.exists():
        log(f"job list missing: {JOBS}; run gen_table5_jobs.py first")
        return 2
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--mem-max", type=int, default=1000,
                    help="strict: max used MiB for a GPU to count as free")
    ap.add_argument("--util-max", type=int, default=5)
    ap.add_argument("--stable", type=int, default=2)
    ap.add_argument("--share", action="store_true", default=False,
                    help="enable share mode fallback (OFF by default: shared GPUs "
                         "require explicit permission; we only take truly idle GPUs)")
    ap.add_argument("--share-after", type=int, default=900,
                    help="seconds without any start before enabling share mode")
    ap.add_argument("--share-min-free", type=int, default=8000)
    ap.add_argument("--share-util", type=int, default=40)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--claim-ttl", type=int, default=3600,
                    help="seconds after which a claim without done/failed is stale")
    ap.add_argument("--max-runtime", type=int, default=0,
                    help="seconds; 0 = unlimited")
    args = ap.parse_args()

    # single instance
    lockdir = QUEUE / "watcher.lock"
    if lockdir.exists():
        pid_file = lockdir / "pid"
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            log(f"another watcher is alive pid={pid}; exit")
            return 0
        except Exception:
            shutil.rmtree(lockdir, ignore_errors=True)
    lockdir.mkdir(parents=True, exist_ok=True)
    (lockdir / "pid").write_text(str(os.getpid()))

    jobs = load_jobs()
    log(f"watcher start pid={os.getpid()} jobs={len(jobs)} host={os.uname().nodename}")
    t0 = time.time()
    streak: dict[int, int] = {}
    workers: dict[int, subprocess.Popen] = {}
    ever_started = False

    while True:
        if args.max_runtime and time.time() - t0 > args.max_runtime:
            log("max runtime reached; exit")
            break
        left = remaining(jobs)
        # reap finished workers
        for gpu in list(workers):
            rc = workers[gpu].poll()
            if rc is not None:
                log(f"worker gpu={gpu} exited rc={rc}")
                del workers[gpu]
                ever_started = True
        if not left and not workers:
            log("ALL JOBS DONE (or failed); exit")
            break

        # stale claim recovery (a worker killed mid-job must not wedge the queue)
        now = time.time()
        for d in LOCKS.iterdir():
            if not d.is_dir():
                continue
            if (LOCKS / f"{d.name}.done").exists() or (LOCKS / f"{d.name}.failed").exists():
                continue
            if now - d.stat().st_mtime > args.claim_ttl:
                log(f"releasing stale claim {d.name} (age > {args.claim_ttl}s)")
                shutil.rmtree(d, ignore_errors=True)

        gpus = query_gpus()
        share = args.share and (not ever_started) and (time.time() - t0 > args.share_after)
        status = {"time": time.strftime("%F %T"), "mode": "share" if share else "strict",
                  "remaining": len(left), "workers": {str(k): v.pid for k, v in workers.items()},
                  "gpus": gpus}
        free_now = []
        for g in gpus:
            if g["index"] in workers or len(workers) >= args.max_workers:
                streak[g["index"]] = 0
                continue
            if share:
                ok = (g["mem_total"] - g["mem_used"] >= args.share_min_free
                      and g["util"] <= args.share_util)
                need = 3
            else:
                ok = (g["mem_used"] < args.mem_max and g["util"] < args.util_max
                      and not g.get("has_compute_app", False))
                need = args.stable
            streak[g["index"]] = streak.get(g["index"], 0) + 1 if ok else 0
            if streak[g["index"]] >= need:
                free_now.append(g["index"])
        status["free_now"] = free_now
        STATUS.write_text(json.dumps(status, indent=1))

        for gpu in free_now:
            if gpu in workers or len(workers) >= args.max_workers:
                continue
            tag = f"gpu{gpu}_{int(time.time())}"
            log(f"GRAB gpu={gpu} mode={'share' if share else 'strict'} tag={tag} remaining={len(left)}")
            w = subprocess.Popen(["bash", str(WORKER), str(gpu), tag],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 cwd=str(REPO), start_new_session=True)
            workers[gpu] = w
            ever_started = True
            streak[gpu] = 0
        time.sleep(args.interval)

    shutil.rmtree(lockdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
