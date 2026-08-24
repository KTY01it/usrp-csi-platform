#!/usr/bin/env python3

import shlex
import subprocess
import time


class RemoteTXController:
    def __init__(
        self,
        host="csi-tx",
        remote_root="~/Documents/usrp-csi-platform/usrp-csi-tx",
        interval_ms=100,
    ):
        self.host = host
        self.remote_root = remote_root
        self.interval_ms = int(interval_ms)

        self.pid_file = "/tmp/csi_sense_tx.pid"
        self.control_log = "/tmp/csi_sense_tx_control.log"

    def _ssh(self, remote_command, check=False):
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                self.host,
                remote_command,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        if check and result.returncode != 0:
            raise RuntimeError(
                "Remote TX command failed:\n"
                + result.stdout
            )

        return result

    def check_connection(self):
        result = self._ssh(
            "echo REMOTE_TX_CONNECTION_PASS"
        )

        return (
            result.returncode == 0
            and "REMOTE_TX_CONNECTION_PASS"
            in result.stdout
        )

    def status(self):
        cmd = f'''
if [ -s {shlex.quote(self.pid_file)} ]; then
    PID=$(cat {shlex.quote(self.pid_file)})

    if kill -0 "$PID" 2>/dev/null; then
        echo "RUNNING $PID"
        exit 0
    fi
fi

echo "STOPPED"
exit 1
'''

        result = self._ssh(cmd)

        text = result.stdout.strip()

        if text.startswith("RUNNING "):
            try:
                pid = int(
                    text.split()[1]
                )
            except Exception:
                pid = None

            return {
                "running": True,
                "pid": pid,
                "raw": text,
            }

        return {
            "running": False,
            "pid": None,
            "raw": text,
        }

    def start(self, startup_wait_s=3.0):
        current = self.status()

        if current["running"]:
            raise RuntimeError(
                "Remote TX is already running "
                f"(PID {current['pid']})"
            )

        interval = int(self.interval_ms)

        remote_root = self.remote_root
        pid_file = shlex.quote(
            self.pid_file
        )
        control_log = shlex.quote(
            self.control_log
        )

        cmd = f'''
set -eu

cd {remote_root}

test -x scripts/run_tx_tdm_2x2.sh

rm -f {pid_file}

nohup setsid env \
    TDM_INTERVAL_MS={interval} \
    TDM_ACTIVE_TX=both \
    ./scripts/run_tx_tdm_2x2.sh \
    > {control_log} 2>&1 \
    < /dev/null &

PID=$!

echo "$PID" > {pid_file}

echo "STARTED $PID"
'''

        result = self._ssh(
            cmd,
            check=True,
        )

        if "STARTED " not in result.stdout:
            raise RuntimeError(
                "Remote TX did not report STARTED:\n"
                + result.stdout
            )

        time.sleep(
            float(startup_wait_s)
        )

        state = self.status()

        if not state["running"]:
            log = self.tail_log(40)

            raise RuntimeError(
                "Remote TX terminated during startup.\n"
                + log
            )

        return state

    def stop(self, shutdown_wait_s=2.0):
        state = self.status()

        if not state["running"]:
            self._ssh(
                f"rm -f {shlex.quote(self.pid_file)}"
            )

            return {
                "stopped": True,
                "was_running": False,
            }

        pid = state["pid"]

        cmd = f'''
PID={int(pid)}

kill -INT -- -"$PID" 2>/dev/null || \
kill -INT "$PID" 2>/dev/null || true
'''

        self._ssh(cmd)

        deadline = (
            time.time()
            + float(shutdown_wait_s)
        )

        while time.time() < deadline:
            if not self.status()["running"]:
                self._ssh(
                    f"rm -f {shlex.quote(self.pid_file)}"
                )

                return {
                    "stopped": True,
                    "was_running": True,
                }

            time.sleep(0.2)

        #
        # Graceful SIGINT failed.
        #
        cmd = f'''
PID={int(pid)}

kill -TERM -- -"$PID" 2>/dev/null || \
kill -TERM "$PID" 2>/dev/null || true

sleep 1

kill -KILL -- -"$PID" 2>/dev/null || \
kill -KILL "$PID" 2>/dev/null || true

rm -f {shlex.quote(self.pid_file)}
'''

        self._ssh(cmd)

        final = self.status()

        return {
            "stopped": not final["running"],
            "was_running": True,
        }

    def tail_log(self, lines=30):
        result = self._ssh(
            "tail -n "
            + str(int(lines))
            + " "
            + shlex.quote(
                self.control_log
            )
            + " 2>/dev/null || true"
        )

        return result.stdout


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=[
            "status",
            "start",
            "stop",
            "log",
            "doctor",
        ],
    )

    parser.add_argument(
        "--interval-ms",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    ctl = RemoteTXController(
        interval_ms=args.interval_ms
    )

    if args.command == "doctor":
        ok = ctl.check_connection()

        print(
            "Remote SSH:",
            "PASS" if ok else "FAIL"
        )

        raise SystemExit(
            0 if ok else 1
        )

    if args.command == "status":
        s = ctl.status()

        print(
            "Remote TX:",
            "RUNNING"
            if s["running"]
            else "STOPPED"
        )

        if s["pid"] is not None:
            print(
                "PID:",
                s["pid"]
            )

        raise SystemExit(0)

    if args.command == "start":
        s = ctl.start()

        print(
            "Remote TX: STARTED"
        )

        print(
            "PID:",
            s["pid"]
        )

        print(
            "Interval:",
            args.interval_ms,
            "ms"
        )

        raise SystemExit(0)

    if args.command == "stop":
        s = ctl.stop()

        print(
            "Remote TX:",
            "STOPPED"
            if s["stopped"]
            else "STOP FAILED"
        )

        raise SystemExit(
            0 if s["stopped"] else 1
        )

    if args.command == "log":
        print(
            ctl.tail_log(50),
            end=""
        )


if __name__ == "__main__":
    main()
