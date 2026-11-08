"""Run the whole experiment on one machine with several GPUs (default 8).

Phase 1: the base model is scored on GPU 0 while book 1 is trained (M1) and M1's own replay/phrasing data
is written on GPU 1. Phase 2: every book-2 run (5 methods x 3 seeds), and the book-3 runs that follow two
of them, are spread over all GPUs, longest first. Logs go to logs/, results to results/.
"""
import os
import queue
import subprocess
import sys
import threading
import time

N = int(os.environ.get("CB_NGPU", "8"))
COST = {"sdft_replay": 10, "naive": 7, "replay": 5, "kl": 4, "sdft": 3}  # rough minutes per chain, for ordering
os.environ["CB_SAVE_M2"] = "1"
os.makedirs("logs", exist_ok=True)
failures = []


def run(gpu, args, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    t0 = time.time()
    with open(log, "a") as f:
        code = subprocess.run([sys.executable, "cb.py"] + args, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"{time.strftime('%H:%M:%S')} gpu{gpu} {' '.join(args)} -> {'ok' if code == 0 else 'FAILED'} in {time.time() - t0:.0f}s", flush=True)
    if code:
        failures.append((args, log))
    return code == 0


def main():
    base = threading.Thread(target=run, args=(0, ["base"], "logs/base.log"))
    base.start()
    if not (run(1, ["stage1", "--seed", "0"], "logs/stage1.log") and run(1, ["prep"], "logs/prep.log")):
        raise SystemExit("book 1 step failed; see logs/stage1.log and logs/prep.log")
    base.join()
    jobs = queue.Queue()
    chains = [(m, s) for s in (0, 1, 2) for m in COST]
    for m, s in sorted(chains, key=lambda c: -COST[c[0]]):
        jobs.put((m, s))

    def worker(gpu):
        while True:
            try:
                m, s = jobs.get_nowait()
            except queue.Empty:
                return
            if run(gpu, ["stage2", "--methods", m, "--seeds", str(s)], f"logs/{m}_s{s}.log") and m in ("naive", "sdft_replay"):
                run(gpu, ["stage3", "--methods", m, "--seeds", str(s)], f"logs/{m}_s{s}.log")

    threads = [threading.Thread(target=worker, args=(g,)) for g in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    subprocess.run([sys.executable, "summarize.py"])
    print("FAILURES", failures if failures else "none")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
