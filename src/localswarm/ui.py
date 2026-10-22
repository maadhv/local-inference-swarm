"""Local browser dashboard and lightweight worker telemetry/control API."""
import json
import socket
import subprocess
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from . import config, gpu, gguf, llama, metrics, network
from .controller import validate_model

FRONTEND = Path(__file__).with_name("frontend") / "index.html"


class WorkerManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.port = config.DEFAULT_RPC_PORT
        self.device = None
        self.log = deque(maxlen=100)

    def start(self, payload):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "Worker RPC service is already running."}
            try:
                port = int(payload.get("rpc_port", config.DEFAULT_RPC_PORT))
                if not 1 <= port <= 65535:
                    raise ValueError("RPC port must be between 1 and 65535.")
                bind = str(payload.get("bind", "0.0.0.0"))
                if not bind or len(bind) > 64:
                    raise ValueError("Invalid bind address.")
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}

            rpc_exe, how = llama.find_executable(config.RPC_SERVER_NAMES, _env(config.ENV_RPC_SERVER))
            if not rpc_exe:
                return {"ok": False, "error": f"RPC server not found: {how}"}
            if not (llama.supports_flag(rpc_exe, "--device") or llama.supports_flag(rpc_exe, "-d")):
                return {"ok": False, "error": "This RPC server does not support device selection; refusing an unsafe fallback."}
            cli, _ = llama.find_executable(config.LLAMA_CLI_NAMES, _env(config.ENV_LLAMA_CLI))
            if not cli:
                return {"ok": False, "error": "llama-cli is needed to select the worker device. Install it or set LOCALSWARM_LLAMA_CLI."}
            devices, raw = llama.list_devices(cli)
            device = gpu.find_nvidia(devices) or llama.Device("CPU", "CPU (system RAM)")
            cmd = [str(rpc_exe), "-H", bind, "-p", str(port), "-d", device.id]
            self.log.clear()
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0)
            except OSError as exc:
                return {"ok": False, "error": f"Could not start RPC server: {exc}"}
            self.proc, self.port, self.device = proc, port, device
            threading.Thread(target=self._read_log, args=(proc,), daemon=True).start()
            time.sleep(0.35)
            if proc.poll() is not None:
                return {"ok": False, "error": "RPC server exited during startup. Check the worker log."}
            return {"ok": True, "message": f"RPC service started on {bind}:{port} using {device.name}.",
                    "device": device.name, "device_id": device.id}

    def _read_log(self, proc):
        try:
            for raw in iter(proc.stdout.readline, b""):
                with self.lock:
                    self.log.append(raw.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):
            pass

    def stop(self):
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                self.proc = None
                return {"ok": True, "message": "Worker RPC service is stopped."}
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with self.lock:
            self.proc = None
        return {"ok": True, "message": "Worker RPC service stopped."}

    def status(self):
        with self.lock:
            running = bool(self.proc and self.proc.poll() is None)
            return {"running": running, "port": self.port, "device": self.device.name if self.device else None,
                    "device_id": self.device.id if self.device else None, "log": list(self.log)[-30:]}


class InferenceManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.proc = None
        self.state = "idle"
        self.log = deque(maxlen=500)
        self.started = None
        self.exit_code = None
        self.error = None

    def start(self, payload):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                return {"ok": False, "error": "Inference is already running."}
        try:
            cmd, plan = _build_inference(payload)
        except (ValueError, OSError, TypeError) as exc:
            return {"ok": False, "error": str(exc)}
        with self.lock:
            self.log.clear()
            self.log.append("Command: " + subprocess.list2cmdline(cmd))
            self.state, self.error, self.exit_code = "starting", None, None
            self.started = time.time()
            try:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, bufsize=0)
            except OSError as exc:
                self.state, self.error = "failed", str(exc)
                return {"ok": False, "error": f"Could not start llama-cli: {exc}"}
            self.proc = proc
            threading.Thread(target=self._run_log, args=(proc,), daemon=True).start()
        return {"ok": True, "message": "Inference started.", "plan": plan}

    def _run_log(self, proc):
        with self.lock:
            self.state = "running"
        try:
            for raw in iter(proc.stdout.readline, b""):
                line = raw.decode("utf-8", "replace").rstrip()
                with self.lock:
                    self.log.append(line)
        except (OSError, ValueError) as exc:
            with self.lock:
                self.log.append(f"Log reader error: {exc}")
        code = proc.wait()
        with self.lock:
            self.exit_code = code
            self.state = "completed" if code == 0 else "failed"

    def stop(self):
        with self.lock:
            proc = self.proc
            if not proc or proc.poll() is not None:
                return {"ok": True, "message": "No inference process is running."}
            self.state = "stopping"
            proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        with self.lock:
            self.state, self.exit_code = "stopped", proc.returncode
        return {"ok": True, "message": "Inference stopped."}

    def status(self):
        with self.lock:
            running = bool(self.proc and self.proc.poll() is None)
            if self.state in ("running", "starting") and not running and self.proc:
                self.state = "completed" if self.proc.returncode == 0 else "failed"
                self.exit_code = self.proc.returncode
            return {"state": self.state, "running": running, "log": list(self.log),
                    "exit_code": self.exit_code, "error": self.error,
                    "elapsed_seconds": round(time.time() - self.started, 1) if self.started else 0}


class DashboardState:
    def __init__(self):
        self.cpu = metrics.CpuSampler()
        self.worker = WorkerManager()
        self.inference = InferenceManager()

    def status(self):
        return {"hostname": socket.gethostname(), "metrics": metrics.system_metrics(self.cpu),
                "worker": self.worker.status(), "inference": self.inference.status()}


def _build_inference(payload):
    model_text = str(payload.get("model", "")).strip()
    model, error = validate_model(model_text)
    if not model:
        raise ValueError(error)
    prompt = str(payload.get("prompt", "")).strip()
    if not prompt:
        raise ValueError("Enter a prompt first.")
    workers = payload.get("workers")
    if not isinstance(workers, list) or not workers:
        raise ValueError("Add at least one worker to the pool.")
    try:
        local_percent = int(payload.get("local_percent", 0))
        remote_percents = [int(w.get("percent", 0)) for w in workers]
        port = int(payload.get("rpc_port", config.DEFAULT_RPC_PORT))
        n_predict = int(payload.get("n_predict", config.DEFAULT_N_PREDICT))
    except (TypeError, ValueError, AttributeError):
        raise ValueError("Pool percentages and ports must be whole numbers.")
    if not 0 <= local_percent <= 100 or any(not 1 <= p <= 100 for p in remote_percents):
        raise ValueError("Controller share must be 0–100%; each worker share must be 1–100%.")
    if local_percent + sum(remote_percents) != 100:
        raise ValueError("Controller and worker layer percentages must add up to 100%.")
    if not 1 <= port <= 65535 or not 1 <= n_predict <= 100000:
        raise ValueError("RPC port or output token limit is out of range.")

    hosts, endpoints = [], []
    for item in workers:
        if not isinstance(item, dict):
            raise ValueError("Invalid worker entry in pool.")
        ip, err = network.validate_ip(str(item.get("host", "")))
        if not ip:
            raise ValueError(err)
        if ip in hosts:
            raise ValueError(f"Worker {ip} appears more than once.")
        hosts.append(ip)
        endpoint = network.rpc_endpoint(ip, int(item.get("rpc_port", port)))
        worker_port = int(item.get("rpc_port", port))
        if not 1 <= worker_port <= 65535:
            raise ValueError(f"Invalid RPC port for worker {ip}.")
        ok, message = network.tcp_check(ip, worker_port)
        if not ok:
            raise ValueError(f"Worker {endpoint} is unreachable: {message}")
        endpoints.append(endpoint)

    cli, how = llama.find_executable(config.LLAMA_CLI_NAMES, _env(config.ENV_LLAMA_CLI))
    if not cli:
        raise ValueError(f"llama-cli not found ({how}). Install an RPC-enabled build or set LOCALSWARM_LLAMA_CLI.")
    if not llama.supports_flag(cli, "--rpc") or not llama.supports_flag(cli, "--device"):
        raise ValueError("llama-cli must support both --rpc and --device.")
    rpc_arg = ",".join(endpoints)
    devices, raw = llama.list_devices(cli, ["--rpc", rpc_arg])
    rpc_devices = [d for d in devices if d.backend.lower() == "rpc"]
    if len(rpc_devices) != len(workers):
        detail = raw.strip()
        raise ValueError(f"Expected one RPC device per worker, found {len(rpc_devices)} for {len(workers)} workers. "
                         "Check compatible RPC builds and that each worker exposes one device. " + detail[-700:])
    try:
        layer_count = gguf.get_layer_count(model)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read GGUF layer count: {exc}")
    remote_percent = sum(remote_percents)
    remote_layers = layer_count if local_percent == 0 else max(1, min(layer_count - 1, round(layer_count * remote_percent / 100)))
    if layer_count < 2:
        raise ValueError("This model needs at least two transformer layers for a split.")
    cmd = [str(cli), "-m", str(model), "--rpc", rpc_arg, "--device", ",".join(d.id for d in rpc_devices),
           "-ngl", str(remote_layers), "-n", str(n_predict)]
    if len(rpc_devices) > 1:
        if not llama.supports_flag(cli, "--tensor-split"):
            raise ValueError("This llama-cli build lacks --tensor-split, which is required for multiple worker shares.")
        cmd += ["--tensor-split", ",".join(str(p) for p in remote_percents)]
    if llama.supports_flag(cli, "--single-turn"):
        cmd.append("-st")
    elif llama.supports_flag(cli, "-no-cnv"):
        cmd.append("-no-cnv")
    cmd += ["-p", prompt]
    plan = {"layers_total": layer_count, "layers_remote": remote_layers,
            "layers_local": layer_count - remote_layers,
            "workers": [{"host": host, "percent": percent, "device": device.name}
                        for host, percent, device in zip(hosts, remote_percents, rpc_devices)],
            "local_percent": local_percent}
    return cmd, plan


def _pool_status(workers):
    if not isinstance(workers, list):
        raise ValueError("Expected a worker list.")
    results = []
    for worker in workers[:32]:
        host, err = network.validate_ip(str(worker.get("host", "")))
        if not host:
            results.append({"host": str(worker.get("host", "")), "online": False, "error": err})
            continue
        try:
            port = int(worker.get("agent_port", 8766))
            if not 1 <= port <= 65535:
                raise ValueError("Invalid UI agent port.")
            url_host = f"[{host}]" if ":" in host else host
            request = Request(f"http://{url_host}:{port}/api/status", headers={"Accept": "application/json"})
            with urlopen(request, timeout=1.5) as response:
                remote = json.loads(response.read(1_000_000).decode("utf-8"))
            result = {"host": host, "online": True, "hostname": remote.get("hostname"),
                      "metrics": remote.get("metrics", {}), "worker": remote.get("worker", {})}
            rpc_port = int(worker.get("rpc_port", config.DEFAULT_RPC_PORT))
            ok, msg = network.tcp_check(host, rpc_port, timeout=1.5)
            result["rpc_online"], result["rpc_message"] = ok, msg
            results.append(result)
        except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            results.append({"host": host, "online": False, "error": str(exc)})
    return results


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def serve_ui(host="127.0.0.1", port=8765, agent_host=None, agent_port=8766):
    state = DashboardState()

    class Handler(BaseHTTPRequestHandler):
        server_version = "LocalSwarmUI/1.0"

        def log_message(self, fmt, *args):
            print("[ui] " + fmt % args)

        def _send(self, value, status=200, content_type="application/json; charset=utf-8"):
            data = value if isinstance(value, bytes) else _json_bytes(value)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > 1_000_000:
                    raise ValueError("Request body too large.")
                return json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid JSON request: {exc}")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                try:
                    page = FRONTEND.read_bytes()
                except OSError as exc:
                    self._send({"error": f"Frontend unavailable: {exc}"}, 500)
                    return
                self._send(page, content_type="text/html; charset=utf-8")
            elif self.path == "/api/status":
                self._send(state.status())
            else:
                self._send({"error": "Not found"}, 404)

        def do_POST(self):
            try:
                payload = self._body()
                if self.path == "/api/worker/start":
                    result = state.worker.start(payload)
                elif self.path == "/api/worker/stop":
                    result = state.worker.stop()
                elif self.path == "/api/pool/status":
                    result = {"workers": _pool_status(payload.get("workers", []))}
                elif self.path == "/api/inference/start":
                    result = state.inference.start(payload)
                elif self.path == "/api/inference/stop":
                    result = state.inference.stop()
                else:
                    self._send({"error": "Not found"}, 404)
                    return
                self._send(result, 200 if result.get("ok", True) else 400)
            except ValueError as exc:
                self._send({"ok": False, "error": str(exc)}, 400)
            except Exception as exc:
                self._send({"ok": False, "error": f"Request failed: {exc}"}, 500)

    class AgentHandler(BaseHTTPRequestHandler):
        """Read-only remote metrics endpoint; never exposes worker control routes."""
        def log_message(self, fmt, *args):
            print("[agent] " + fmt % args)

        def do_GET(self):
            if self.path != "/api/status":
                data = _json_bytes({"error": "Not found"})
                self.send_response(404)
            else:
                data = _json_bytes(state.status())
                self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            data = _json_bytes({"error": "The telemetry agent is read-only."})
            self.send_response(405)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as exc:
        print(f"Could not start dashboard on {host}:{port}: {exc}")
        return 1
    server.daemon_threads = True
    print(f"LocalSwarm dashboard: http://127.0.0.1:{port}")
    agent = None
    if agent_host:
        try:
            agent = ThreadingHTTPServer((agent_host, agent_port), AgentHandler)
        except OSError as exc:
            server.server_close()
            print(f"Could not start telemetry agent on {agent_host}:{agent_port}: {exc}")
            return 1
        agent.daemon_threads = True
        threading.Thread(target=agent.serve_forever, daemon=True).start()
        print(f"Read-only worker telemetry agent: {agent_host}:{agent_port}")
        if agent_host not in ("127.0.0.1", "localhost", "::1"):
            print("The agent exposes hardware metrics without authentication; use only on a trusted LAN.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("Dashboard control routes are exposed on the network without authentication. Avoid this unless needed.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        state.inference.stop()
        state.worker.stop()
        server.server_close()
        if agent:
            agent.shutdown()
            agent.server_close()
    return 0


def _env(name):
    import os
    return os.environ.get(name)
