"""Controller / Prompting Host: validates everything, then runs llama-cli with --rpc."""
import codecs
import re
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from . import config, gguf, llama, network

BAR = "=" * 60


# ---------- input helpers ----------
def ask_ip(preset=None):
    while True:
        raw = preset or input("Worker IP: ")
        ip, err = network.validate_ip(raw)
        if ip:
            return ip
        print(f"[!] {err}")
        preset = None


def validate_model(text: str):
    p = Path(text.strip().strip('"').strip("'"))
    if not p.exists():
        return None, f"File does not exist: {p}"
    if p.is_dir():
        ggufs = sorted(p.glob("*.gguf"))
        hint = f"\n    Did you mean: {ggufs[0]}" if ggufs else "\n    Point to the .gguf FILE inside it."
        return None, f"That is a directory (e.g. a snapshot folder), not a model file: {p}{hint}"
    if p.suffix.lower() != ".gguf":
        return None, f"Not a .gguf file (extension is '{p.suffix}'): {p}"
    return p.resolve(), None


def ask_model(preset=None):
    while True:
        raw = preset or input("GGUF Model Path: ")
        model, err = validate_model(raw)
        if model:
            return model
        print(f"[!] {err}")
        preset = None


# ---------- inference runner ----------
_PERF_NEW = re.compile(r"\[\s*Prompt:\s*([\d.]+)\s*t/s\s*\|\s*Generation:\s*([\d.]+)\s*t/s\s*\]")
_PERF_OLD = re.compile(r"eval time\s*=.*?/\s*(\d+)\s*runs.*?([\d.]+)\s*tokens per second")
_MODEL_LOADED = re.compile(r"load_tensors:\s+offloaded|llama_context:|model loaded", re.I)
_RPC_BUFFER = re.compile(r"\bRPC\w*\b.*buffer size", re.I)
_FAILURES = [
    (re.compile(r"RPC not supported", re.I), "This llama-cli build does not support RPC."),
    (re.compile(r"ErrorOutOfDeviceMemory|out of (device )?memory|failed to allocate|allocation of size .* failed", re.I),
     "Out of memory (Vulkan/host). Use a smaller model/context, or check which device is being used."),
    (re.compile(r"failed to load model|error loading model|unable to load model", re.I),
     "The model failed to load (corrupt/incompatible GGUF or not enough memory)."),
    (re.compile(r"(rpc.*(fail|error|refused|disconnect))|failed to connect|connection (reset|closed|refused)|send failed|recv failed", re.I),
     "RPC connection problem: the worker may have disconnected or crashed mid-run."),
]


class Result:
    def __init__(self):
        self.returncode = None
        self.interrupted = False
        self.stdout = ""
        self.stderr_tail = deque(maxlen=60)
        self.model_loaded = False
        self.first_output = False
        self.prompt_tps = None
        self.gen_tps = None
        self.gen_tokens = None
        self.rpc_buffer_seen = False
        self.failure_hint = None


def _scan_perf(text: str, r: Result):
    m = _PERF_NEW.search(text)
    if m:
        r.prompt_tps, r.gen_tps = float(m.group(1)), float(m.group(2))
    m = _PERF_OLD.findall(text)
    if m and r.gen_tps is None:
        runs, tps = m[-1]
        r.gen_tokens, r.gen_tps = int(runs), float(tps)


def run_inference(cmd, verbose=False) -> Result:
    r = Result()
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
    except OSError as e:
        r.returncode = -1
        r.stderr_tail.append(f"Could not start llama-cli: {e}")
        return r

    out_chunks = []

    def read_out():
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            b = proc.stdout.read(256)
            if not b:
                break
            s = dec.decode(b)
            if s:
                if not r.first_output:
                    r.first_output = True
                    print("[+] Inference started -- model output follows\n" + "-" * 60, flush=True)
                out_chunks.append(s)
                sys.stdout.write(s)
                sys.stdout.flush()

    def read_err():
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            r.stderr_tail.append(line)
            if verbose:
                print(f"[llama] {line}", flush=True)
            if not r.model_loaded and _MODEL_LOADED.search(line):
                r.model_loaded = True
                print("[+] Model loaded", flush=True)
            if _RPC_BUFFER.search(line):
                r.rpc_buffer_seen = True
            _scan_perf(line, r)
            if r.failure_hint is None:
                for rx, msg in _FAILURES:
                    if rx.search(line):
                        r.failure_hint = msg
                        break

    threads = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=read_err, daemon=True)]
    for t in threads:
        t.start()
    try:
        while proc.poll() is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        r.interrupted = True
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    for t in threads:
        t.join(timeout=3)
    r.returncode = proc.returncode
    r.stdout = "".join(out_chunks)
    _scan_perf(r.stdout, r)
    return r


# ---------- main flow ----------
def run_controller(args) -> int:
    print(f"{BAR}\n{'PROMPTING HOST':^60}\n{BAR}\n")
    try:
        ip = ask_ip(args.ip)
        if args.list_devices:
            return _list_devices(args, ip)
        model = ask_model(args.model)
        prompt = args.prompt or input("Prompt: ").strip()
        if not prompt:
            print("[!] Empty prompt.")
            return 1
        if not 1 <= args.split_percent <= 99:
            print("[!] --split-percent must be between 1 and 99.")
            return 1
    except EOFError:
        print("\n[!] Input closed.")
        return 1
    except KeyboardInterrupt:
        print("\n[!] Cancelled.")
        return 130
    print()

    cli, how = llama.find_executable(config.LLAMA_CLI_NAMES, args.llama_cli or _env(config.ENV_LLAMA_CLI))
    if not cli:
        print(f"[!] llama-cli not found: {how}")
        print("    Install llama.cpp (winget install ggml.llamacpp) or pass --llama-cli <path>.")
        return 1
    print(f"[✓] llama-cli: {cli}  (via {how})")
    print(f"    {llama.get_version(cli)}")

    if not llama.supports_flag(cli, "--rpc"):
        print("\n[!] This llama-cli does NOT support --rpc. Not attempting inference.")
        print(f"    Executable: {cli}\n    Version:    {llama.get_version(cli)}")
        print("    This build was compiled without RPC (typical of an old copied llama-cli.exe, which gives\n"
              "    'RPC not supported in this build'). Remove it from PATH/cwd, or pass --llama-cli <path> "
              "to the WinGet install.")
        return 1
    if not llama.supports_flag(cli, "--device"):
        print("[!] This llama-cli build does not support --device, so the local integrated GPU cannot be excluded safely.")
        print("    Install a newer RPC-enabled llama.cpp build and try again.")
        return 1
    print("[✓] RPC support detected")
    print(f"[✓] GGUF validated: {model}")

    port = args.port
    ok, msg = network.tcp_check(ip, port)
    if not ok:
        print(f"\n[!] Worker not reachable at {ip}:{port}\n    {msg}")
        print("    Check: worker started first? same network? Windows Firewall on the worker allows "
              f"inbound TCP {port}? (This tool does not open firewall rules.)")
        return 1
    print("[✓] Worker reachable (TCP only -- not proof the RPC protocol/GPU works)")
    print(f"[✓] RPC endpoint: {ip}:{port}")

    devices, raw_devices = llama.list_devices(cli, ["--rpc", f"{ip}:{port}"])
    rpc_device = next((d for d in devices if d.backend.lower() == "rpc"), None)
    if not rpc_device:
        print("[!] llama.cpp did not report a remote RPC device. Cannot safely choose the worker for the split.")
        if raw_devices.strip():
            print("    Devices reported:\n" + raw_devices.strip())
        print("    Confirm that the worker is running an RPC-enabled llama.cpp build and that both builds are compatible.")
        return 1
    try:
        total_layers = gguf.get_layer_count(model)
    except (OSError, ValueError) as e:
        print(f"[!] Could not read the model's layer count from its GGUF metadata: {e}")
        print("    Use a valid GGUF file. The 50/50 layer split needs the model's block_count metadata.")
        return 1
    remote_layers = args.gpu_layers if args.gpu_layers is not None else max(
        1, min(total_layers - 1, round(total_layers * args.split_percent / 100)))
    if not 1 <= remote_layers < total_layers:
        print(f"[!] Worker layer count must be from 1 to {total_layers - 1} for a two-sided split.")
        return 1
    local_layers = total_layers - remote_layers
    print(f"[✓] Worker device: {rpc_device.id} ({rpc_device.name})")
    print(f"[✓] Layer split: {remote_layers}/{total_layers} on worker; {local_layers}/{total_layers} on controller CPU/RAM")

    cmd = [str(cli), "-m", str(model), "--rpc", f"{ip}:{port}", "--device", rpc_device.id,
       "-ngl", str(remote_layers), "-n", str(args.n_predict)]
    if llama.supports_flag(cli, "--single-turn"):
        cmd += ["-st"]
    elif llama.supports_flag(cli, "-no-cnv"):
        cmd += ["-no-cnv"]
    cmd += ["-p", prompt]

    print("[✓] Controller local GPU excluded; remaining layers stay on controller CPU/RAM")
    print("\nCommand:\n  " + " ".join(f'"{c}"' if (" " in c or not c) else c for c in cmd))
    print("\n[+] Starting inference...  (Ctrl+C to interrupt)\n")

    r = run_inference(cmd, verbose=args.verbose)
    print("\n" + "-" * 60)
    return _report(r, args)


def _report(r: Result, args) -> int:
    if r.interrupted:
        print("[!] Inference interrupted by user.")
        return 130
    if r.returncode != 0:
        print(f"[!] Inference failed. llama-cli exit code: {r.returncode}")
        if r.failure_hint:
            print(f"    Likely cause: {r.failure_hint}")
        print("    Last llama-cli log lines:")
        for ln in list(r.stderr_tail)[-25:]:
            print(f"      {ln}")
        return 1
    produced = bool(r.stdout.strip()) or r.gen_tps is not None
    if not produced:
        print("[!] llama-cli exited with code 0 but produced no output. NOT reporting success.")
        for ln in list(r.stderr_tail)[-15:]:
            print(f"      {ln}")
        return 1

    print("[+] Inference completed (exit code 0, output generated)")
    if r.gen_tps is not None:
        pp = f"Prompt: {r.prompt_tps} t/s | " if r.prompt_tps is not None else ""
        print(f"    [ {pp}Generation: {r.gen_tps} t/s ]")
    else:
        print("    (no llama.cpp throughput line found in this build's output)")
    if r.rpc_buffer_seen:
        print("    Log evidence: llama.cpp placed buffers on an RPC device (see --verbose log).")
    else:
        print("[!] No RPC buffer allocation seen in llama.cpp's log. The model may have run entirely on the "
              "controller CPU; check the --verbose logs and worker activity.")
    print("    Confirm worker compute from its GPU monitor (NVIDIA: `nvidia-smi -l 1`) or CPU activity.")
    return 0


def _list_devices(args, ip) -> int:
    cli, how = llama.find_executable(config.LLAMA_CLI_NAMES, args.llama_cli or _env(config.ENV_LLAMA_CLI))
    if not cli:
        print(f"[!] llama-cli not found: {how}")
        return 1
    ok, msg = network.tcp_check(ip, args.port)
    if not ok:
        print(f"[!] {msg}")
        return 1
    print(f"llama-cli: {cli}\nDevices visible with --rpc {ip}:{args.port}:")
    devs, raw = llama.list_devices(cli, ["--rpc", f"{ip}:{args.port}"])
    print(raw.strip() if not devs else "\n".join(f"  {d.id}: {d.name}" for d in devs))
    return 0


def _env(name):
    import os
    return os.environ.get(name)
