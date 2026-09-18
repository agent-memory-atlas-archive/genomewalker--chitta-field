"""Replica WAL soak; run from the parent chitta checkout on a compute node.

python3 chitta-field/tests/wal_replica_soak.py --output /scratch/unique-run
Defaults enforce the 30-minute, 20-Hz proof with compaction every minute.
The frozen source is copied by eval-replica.sh; no live daemon is used.
"""

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import shlex
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=pathlib.Path)
parser.add_argument("--port", default=18421, type=int)
parser.add_argument(
    "--frozen",
    type=pathlib.Path,
    default=pathlib.Path("/projects/caeg/scratch/kbd606/tmp/learning-cut-20260915-frozen"),
)
args = parser.parse_args()
root = args.output.resolve()
repo = pathlib.Path(__file__).resolve().parents[2]
os.chdir(repo)
if (root / "mind").exists():
    parser.error("output already contains a mind; choose a fresh output directory")
root.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
for key in ("CHITTA_GPU_AUTOSTART", "CHITTA_EMBED_GPU_ONLY"):
    env.pop(key, None)
env["CUDA_VISIBLE_DEVICES"] = ""
env.update(
    CHITTA_EVAL_MIND=str(root / "mind"),
    CHITTA_LIVE_MIND=str(args.frozen.resolve()),
    CHITTA_EVAL_PORT=str(args.port),
    CHITTAD_BIN=str(repo / "bin/chittad"),
    CHITTA_BIN=str(repo / "bin/chitta"),
    CHITTA_RECALL_NOW="1789516800000",
    CHITTA_RECALL_EMBED_WAIT_MS="10000",
    OPENBLAS_NUM_THREADS="1",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    CHITTA_NO_QUEUE="1",
)


def replica(action):
    subprocess.run(["bash", "scripts/eval-replica.sh", action], env=env, check=True, timeout=900)


def rpc(tool, **args):
    req = dict(jsonrpc="2.0", id=1, method="tools/call", params=dict(name=tool, arguments=args))
    result = subprocess.run(
        [env["CHITTA_BIN"], "--socket-path", sock],
        input=json.dumps(req) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-1000:]
    resp = json.loads(result.stdout)
    assert "error" not in resp, resp
    result = resp["result"]
    assert not result.get("isError"), result
    return result.get("structured", result)


def count():
    return rpc("health_check")["memory_count"]


def write(i):
    ans = rpc(
        "remember",
        content=f"p21 WAL durability sample {root.name} record {i:06d}",
        realm="p21-wal-soak",
        type="wisdom",
    )
    mid = ans.get("id", ans.get("memory_id"))
    assert mid is not None, ans
    return str(mid)


report = dict(status="FAIL", planned_writes=36000, duration_target_s=1800, issued_hz=20)
try:
    replica("start")
    values = {}
    for line in (root / "mind/replica.env").read_text().splitlines():
        k, v = line.split("=", 1)
        values[k] = shlex.split(v)[0]
    sock = values["CHITTA_EVAL_SOCKET"]
    initial = count()
    samples = [(0, initial)]
    writes = []
    compactions = []
    begin = time.monotonic()
    with (
        cf.ThreadPoolExecutor(max_workers=16) as writers,
        cf.ThreadPoolExecutor(max_workers=1) as maintenance,
    ):
        for i in range(36000):
            delay = begin + i / 20 - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            writes.append(writers.submit(write, i))
            if i % 1200 == 0:
                compactions.append(maintenance.submit(rpc, "compact_wal"))
                print(
                    json.dumps(
                        dict(
                            elapsed_s=round(time.monotonic() - begin, 2),
                            issued=i + 1,
                            acknowledged=sum(f.done() and f.exception() is None for f in writes),
                            memory_count=samples[-1][1],
                            compact_issued=len(compactions),
                        )
                    ),
                    flush=True,
                )
            if i % 20 == 0:
                samples.append((time.monotonic() - begin, count()))
        delay = begin + 1800 - time.monotonic()
        if delay > 0:
            time.sleep(delay)
    completed_s = time.monotonic() - begin
    ids = [f.result() for f in writes]
    compact_results = [f.result() for f in compactions]
    samples.append((time.monotonic() - begin, count()))
    for idx in (0, len(ids) // 2, len(ids) - 1):
        assert f"record {idx:06d}" in json.dumps(rpc("get", id=ids[idx]))
    log = (root / "mind/replica.log").read_text(errors="replace")
    vanished = len(re.findall(r"WAL segment[^\n]*vanished", log, re.I))
    stale = len(re.findall(r"ESTALE|Stale file handle|os error 116", log, re.I))
    regressions = sum(b[1] < a[1] for a, b in zip(samples, samples[1:]))
    report.update(
        acknowledged_hz=round(len(ids) / completed_s, 4),
        elapsed_s=round(time.monotonic() - begin, 3),
        acknowledged=len(ids),
        unique_ids=len(set(ids)),
        compactions=len(compact_results),
        compact_results=compact_results,
        initial_memory_count=initial,
        final_memory_count=samples[-1][1],
        count_samples=len(samples),
        count_regressions=regressions,
        vanished_lines=vanished,
        estale_lines=stale,
    )
    (root / "samples.json").write_text(json.dumps(samples))
    assert len(ids) == len(set(ids)) == 36000, report
    assert len(ids) / completed_s >= 19.9, report
    assert len(compact_results) == 30 and not regressions and not vanished and not stale, report
    assert samples[-1][1] >= initial + 36000, report
    report["status"] = "PASS"
except BaseException as error:
    report["error"] = repr(error)
    raise
finally:
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "compact_results"}), flush=True)
    replica("stop")
