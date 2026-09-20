#!/usr/bin/env python3
import json
import os
import subprocess
import time
from datetime import datetime

LOG_PATH = "/home/ubuntu/upload_git_watchdog.log"
CHECKPOINT_PATH = "/home/ubuntu/rsna_git_upload_checkpoint.json"
UPLOAD_SCRIPT = "/home/ubuntu/upload_missing_git.py"


def log(msg):
    with open(LOG_PATH, "a") as f:
        f.write(f"{datetime.utcnow().isoformat()}Z {msg}\n")


def get_total_done():
    try:
        with open(CHECKPOINT_PATH) as f:
            return json.load(f).get("total_done", -1)
    except Exception:
        return -1


def main():
    log("Git upload watchdog started")
    proc = None
    failures = []
    last_done = -1
    last_done_time = time.time()
    env = os.environ.copy()
    env.setdefault("HF_HUB_DISABLE_XET", "1")

    while True:
        if proc is None or proc.poll() is not None:
            if proc is not None:
                rc = proc.returncode
                if rc == 0:
                    log("Upload script exited successfully")
                    break
                log(f"Upload script exited with code {rc}; will restart")
                failures.append(time.time())
                failures = [t for t in failures if time.time() - t < 600]
                if len(failures) >= 5:
                    log("Too many failures in 10 minutes; stopping watchdog")
                    break
            log("Starting upload script")
            proc = subprocess.Popen(
                ["python3", "-u", UPLOAD_SCRIPT],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            done = get_total_done()
            if done != last_done:
                last_done = done
                last_done_time = time.time()
            elif time.time() - last_done_time > 1800:
                log(f"Checkpoint stuck at total_done={last_done} for 30 minutes; restarting upload")
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.kill()
                except Exception:
                    pass
                proc = None
        time.sleep(30)


if __name__ == "__main__":
    main()
