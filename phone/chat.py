#!/usr/bin/env python3
"""Interactive terminal chat interface for GGUF models on mobile/edge devices.

Features:
- Standard library only (no external dependencies)
- Launches and manages llama-server subprocess
- Streaming response generation via HTTP API
- Live status line pinned to bottom terminal row (device RAM, model RAM, model CPU)
- Monitors resource usage via /proc before loading, while running, and after unload
- Safe shutdown on exit, Ctrl+C, or unexpected errors

Usage:
    python phone/chat.py path/to/model.gguf [--port 8080] [--host 127.0.0.1]
"""

import argparse
import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Tuple


def get_terminal_size() -> Tuple[int, int]:
    """Get current terminal columns and rows."""
    cols, rows = shutil.get_terminal_size(fallback=(80, 24))
    return cols, rows


def read_device_memory() -> Dict[str, float]:
    """Read system-wide RAM metrics in MB from /proc/meminfo."""
    mem: Dict[str, float] = {"total_mb": 0.0, "available_mb": 0.0, "used_mb": 0.0}
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                val_str = parts[1].strip().split()[0]
                data[key] = float(val_str) / 1024.0  # kB to MB
        mem["total_mb"] = data.get("MemTotal", 0.0)
        mem["available_mb"] = data.get("MemAvailable", 0.0)
        mem["used_mb"] = mem["total_mb"] - mem["available_mb"]
    except Exception:
        pass
    return mem


def read_process_stats(pid: int) -> Dict[str, float]:
    """Read process RSS (RAM) in MB and CPU ticks from /proc/<pid>/."""
    stats = {"rss_mb": 0.0, "utime": 0.0, "stime": 0.0}
    try:
        # Read RSS from status
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    stats["rss_mb"] = float(parts[1]) / 1024.0  # kB to MB
                    break
        # Read CPU ticks from stat
        with open(f"/proc/{pid}/stat", "r") as f:
            fields = f.read().split()
            # utime is 14th field (index 13), stime is 15th (index 14)
            if len(fields) >= 15:
                stats["utime"] = float(fields[13])
                stats["stime"] = float(fields[14])
    except Exception:
        pass
    return stats


class ChatSession:
    """Manages the llama-server lifecycle, chat loop, and status rendering."""

    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.model_path = os.path.abspath(model_path)
        self.host = host
        self.port = port
        self.server_proc: Optional[subprocess.Popen] = None
        self._prev_proc_cpu = 0.0
        self._prev_time = time.time()
        self._cpu_percent = 0.0

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

    def print_resource_summary(self, stage: str) -> None:
        """Print device and process resource usage summary."""
        mem = read_device_memory()
        print(f"\n--- Resource Usage [{stage}] ---")
        if mem["total_mb"] > 0:
            print(f"  Device RAM: {mem['used_mb']:.1f} MB / {mem['total_mb']:.1f} MB used "
                  f"({(mem['used_mb'] / mem['total_mb'] * 100):.1f}%)")
        if self.server_proc and self.server_proc.poll() is None:
            p_stats = read_process_stats(self.server_proc.pid)
            print(f"  Model Server PID: {self.server_proc.pid} | VmRSS: {p_stats['rss_mb']:.1f} MB")
        print("---------------------------------")

    def start_server(self) -> None:
        """Start llama-server in the background and wait for readiness."""
        self.print_resource_summary("Before Loading")
        print(f"Starting llama-server for: {os.path.basename(self.model_path)}...")

        cmd = [
            "llama-server",
            "-m", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "-c", "2048",
            "--log-disable",
        ]

        try:
            self.server_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            # Fallback check if llama-server is not in PATH
            print("Warning: llama-server executable not found in PATH.")
            print("Assuming external server or mock mode for chat.")
            return

        # Poll health endpoint until server is ready
        health_url = f"http://{self.host}:{self.port}/health"
        print("Waiting for model to load into memory...")
        start_wait = time.time()
        ready = False
        while time.time() - start_wait < 120:
            if self.server_proc.poll() is not None:
                raise RuntimeError("llama-server terminated unexpectedly during loading.")
            try:
                req = urllib.request.Request(health_url)
                with urllib.request.urlopen(req, timeout=1.0) as resp:
                    if resp.status == 200:
                        ready = True
                        break
            except Exception:
                time.sleep(0.5)

        if not ready:
            self.stop_server()
            raise TimeoutError("llama-server timed out during startup.")

        print("Model loaded successfully!")
        self.print_resource_summary("After Loading")

    def stop_server(self) -> None:
        """Stop llama-server subprocess gracefully."""
        if self.server_proc is not None and self.server_proc.poll() is None:
            print("\nShutting down llama-server...")
            self.server_proc.terminate()
            try:
                self.server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.server_proc.kill()
            self.server_proc = None
        self.print_resource_summary("After Unloading")

    def update_cpu_share(self) -> float:
        """Estimate server CPU utilization percentage since last call."""
        if not self.server_proc or self.server_proc.poll() is not None:
            return 0.0
        p_stats = read_process_stats(self.server_proc.pid)
        curr_proc_cpu = p_stats["utime"] + p_stats["stime"]
        curr_time = time.time()
        dt = curr_time - self._prev_time
        if dt > 0.2:
            # Clock ticks per second (usually 100 on Linux)
            clk_tck = os.sysconf(os.sysconf_names.get("SC_CLK_TCK", "SC_CLK_TCK")) if hasattr(os, "sysconf") else 100.0
            ticks_delta = curr_proc_cpu - self._prev_proc_cpu
            cpu_secs = ticks_delta / float(clk_tck)
            self._cpu_percent = min(100.0, max(0.0, (cpu_secs / dt) * 100.0))
            self._prev_proc_cpu = curr_proc_cpu
            self._prev_time = curr_time
        return self._cpu_percent

    def render_status_bar(self) -> None:
        """Pin a live status bar at the bottom row of the terminal."""
        cols, rows = get_terminal_size()
        dev_mem = read_device_memory()
        cpu_pct = self.update_cpu_share()

        model_rss = 0.0
        if self.server_proc and self.server_proc.poll() is None:
            p_stats = read_process_stats(self.server_proc.pid)
            model_rss = p_stats["rss_mb"]

        status = (
            f" [RAM: {dev_mem['used_mb']:.0f}/{dev_mem['total_mb']:.0f}MB] "
            f"[Model: {model_rss:.0f}MB] "
            f"[CPU: {cpu_pct:.1f}%] "
        )
        status_line = status.ljust(cols)[:cols]

        # ANSI: save cursor, move to bottom row, write inverse video bar, restore cursor
        sys.stdout.write(f"\033[s\033[{rows};1H\033[7m{status_line}\033[0m\033[u")
        sys.stdout.flush()

    def stream_completion(self, messages: list) -> str:
        """Send chat request and stream chunks to stdout."""
        url = f"http://{self.host}:{self.port}/v1/chat/completions"
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 512,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        full_response = []
        try:
            with urllib.request.urlopen(req) as resp:
                for raw_line in resp:
                    self.render_status_bar()
                    line = raw_line.decode("utf-8").strip()
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line_content = line[6:].strip()
                        if line_content == "[DONE]":
                            break
                        try:
                            chunk = json.loads(line_content)
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                full_response.append(content)
                        except json.JSONDecodeError:
                            continue
        except urllib.error.URLError as e:
            sys.stdout.write(f"\n[Error communicating with server: {e}]\n")

        sys.stdout.write("\n")
        sys.stdout.flush()
        return "".join(full_response)

    def run(self) -> None:
        """Main interaction loop."""
        self.start_server()
        print("\nType your message and press Enter (or 'exit' / Ctrl+C to quit).\n")
        messages = [{"role": "system", "content": "You are a helpful and concise assistant."}]

        try:
            while True:
                self.render_status_bar()
                try:
                    user_input = input("User > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break

                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit", ":q"):
                    break

                messages.append({"role": "user", "content": user_input})
                sys.stdout.write("Assistant > ")
                sys.stdout.flush()

                reply = self.stream_completion(messages)
                messages.append({"role": "assistant", "content": reply})
        finally:
            self.stop_server()


def main() -> None:
    parser = argparse.ArgumentParser(description="quantlab phone chat client")
    parser.add_argument("model_path", help="Path to .gguf model file")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default 8080)")
    args = parser.parse_args()

    session = ChatSession(args.model_path, host=args.host, port=args.port)

    # Register signal and atexit handlers
    def sig_handler(sig, frame):
        session.stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    atexit.register(session.stop_server)

    session.run()


if __name__ == "__main__":
    main()
