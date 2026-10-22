"""Worker / Inference Host: exposes NVIDIA GPU when present, otherwise CPU/RAM."""
import subprocess
import sys
import threading
import time
from collections import deque

from . import config, gpu, llama

BAR = "=" * 60


def run_worker(args) -> int:
    print(f"{BAR}\n{'WORKER NODE':^60}\n{BAR}\n")

    print("Detecting llama.cpp...")
    rpc_exe, how = llama.find_executable(config.RPC_SERVER_NAMES, args.rpc_server or _env(config.ENV_RPC_SERVER))
    if not rpc_exe:
        print(f"[!] RPC server executable not found: {how}")
        print("    Install llama.cpp (winget install ggml.llamacpp) or pass --rpc-server <path>.")
        return 1
    print(f"[✓] rpc-server: {rpc_exe}  (via {how})")
    print(f"    {llama.get_version(rpc_exe)}")

    # -d / --device must exist so integrated GPUs can be excluded on CPU fallback.
    if not (llama.supports_flag(rpc_exe, "--device") or llama.supports_flag(rpc_exe, "-d")):
        print("[!] This rpc-server build does not advertise -d/--device, so compute cannot be pinned safely.")
        print("    Refusing to start because it could silently select an integrated GPU. Update llama.cpp.")
        return 1

    print("Detecting GPU...")
    device = None
    cli, cli_how = llama.find_executable(config.LLAMA_CLI_NAMES, args.llama_cli or _env(config.ENV_LLAMA_CLI))
    devices = []
    if cli:
        print(f"[✓] llama-cli (for device detection): {cli}  (via {cli_how})")
        devices, raw = llama.list_devices(cli)
        if not devices:
            print("[!] Could not parse `llama-cli --list-devices` output:\n" + raw.strip())
    else:
        print(f"[!] llama-cli not found ({cli_how}); cannot auto-detect devices.")
        if not args.device:
            print("    Install an RPC-enabled llama.cpp CLI, or pass --device <ID> to select the worker device explicitly.")
            return 1

    if devices:
        print("    Devices reported by llama.cpp:")
        for d in devices:
            mem = f", {d.mem_mib} MiB" if d.mem_mib else ""
            print(f"      {d.id}: {d.name}{mem}")

    if args.device:
        device = next((d for d in devices if d.id.lower() == args.device.lower()), None) \
            or llama.Device(args.device, f"(user-specified {args.device})")
        print(f"[!] Using device from --device: {device.id} (not auto-verified as NVIDIA)"
              if not any(d.id == device.id for d in devices) else f"    Using device from --device: {device.id}")
    else:
        device = gpu.find_nvidia(devices)
        if not device:
            # Pin CPU explicitly so a weak integrated GPU is not chosen as an accelerator.
            device = llama.Device("CPU", "CPU (system RAM)")
            print("[!] No NVIDIA device found. Exposing the CPU/RAM backend; integrated GPUs are not selected.")

    print(f"\n[WORKER]\nCompute device: {device.name}\nBackend: {device.backend}\nDevice: {device.id}")
    print(f"RPC: {args.bind}:{args.port}\nStatus: STARTING\n")
    if args.bind == "0.0.0.0":
        print("[!] Bound to 0.0.0.0 for the LAN test. Do not expose this port to the internet; "
              "no firewall rule is opened automatically.\n")

    cmd = [str(rpc_exe), "-H", args.bind, "-p", str(args.port), "-d", device.id]
    print("Starting RPC server...\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, bufsize=0)
    except OSError as e:
        print(f"[!] Failed to start RPC server: {e}")
        return 1

    tail = deque(maxlen=40)
    stop = threading.Event()

    def pump():
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            tail.append(line)
            print(f"[RPC] {line}", flush=True)
            low = line.lower()
            if "accepted client connection" in low:
                print("[WORKER] Status: CONNECTED -- RPC connection only; this is NOT proof of compute. "
                      "Watch GPU activity (for NVIDIA, `nvidia-smi -l 1`) during generation.", flush=True)
            elif "client connection closed" in low or "client disconnected" in low:
                print("[WORKER] Status: WAITING (controller disconnected)", flush=True)
        stop.set()

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    time.sleep(1.5)
    if proc.poll() is not None:
        t.join(timeout=2)
        print(f"[!] RPC server exited immediately (code {proc.returncode}).")
        _explain_worker_failure("\n".join(tail), args.port)
        return 1

    print(f"\n[WORKER]\nCompute device: {device.name}\nBackend: {device.backend}\nRPC: {args.bind}:{args.port}\nStatus: WAITING")
    print("\nWaiting for controller... (Ctrl+C to stop)")
    print("Verify GPU use from another PowerShell window:  nvidia-smi -l 1\n")

    try:
        while proc.poll() is None:
            time.sleep(0.5)
        t.join(timeout=2)
        print(f"\n[!] RPC server stopped unexpectedly (exit code {proc.returncode}).")
        _explain_worker_failure("\n".join(tail), args.port)
        return 1
    except KeyboardInterrupt:
        print("\n[!] Stopping worker...")
        return 0
    finally:
        _terminate(proc)


def _explain_worker_failure(log: str, port: int):
    low = log.lower()
    if "address already in use" in low or "bind" in low and "fail" in low:
        print(f"    Port {port} is probably already in use (another rpc-server still running?).")
    if "vulkan" in low and ("out of" in low and "memory" in low or "allocate" in low):
        print("    Vulkan reported a memory allocation problem on the selected GPU.")
    if "unknown argument" in low or "invalid" in low and "argument" in low:
        print("    This rpc-server build rejected the command-line arguments.")


def _terminate(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _env(name):
    import os
    return os.environ.get(name)
