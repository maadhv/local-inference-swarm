"""Entry point: menu or subcommands (`worker`, `controller`)."""
import argparse
import sys

from . import config
from .controller import run_controller
from .worker import run_worker

BAR = "=" * 60


def build_parser():
    p = argparse.ArgumentParser(prog="localswarm", description="LocalSwarm MVP: llama.cpp RPC orchestration")
    sub = p.add_subparsers(dest="mode")

    w = sub.add_parser("worker", help="Inference host: run ggml-rpc-server on the NVIDIA GPU")
    w.add_argument("--port", type=int, default=config.DEFAULT_RPC_PORT)
    w.add_argument("--bind", default=config.DEFAULT_BIND, help="bind address (default 0.0.0.0, LAN test only)")
    w.add_argument("--device", help="force a llama.cpp device id (e.g. Vulkan1); default: auto-detect NVIDIA")
    w.add_argument("--rpc-server", help="path to ggml-rpc-server.exe")
    w.add_argument("--llama-cli", help="path to llama-cli.exe (used only for device detection)")

    c = sub.add_parser("controller", help="Prompting host: send a prompt through llama.cpp RPC")
    c.add_argument("--ip", help="worker IP")
    c.add_argument("--model", help="path to local .gguf file")
    c.add_argument("--prompt")
    c.add_argument("--port", type=int, default=config.DEFAULT_RPC_PORT)
    c.add_argument("--llama-cli", help="path to llama-cli.exe")
    c.add_argument("-n", "--n-predict", type=int, default=config.DEFAULT_N_PREDICT)
    c.add_argument("--local-device", default=config.LOCAL_DEVICE_DEFAULT,
                   help="value for llama-cli --device (default 'none': no local GPU)")
    c.add_argument("--gpu-layers", type=int, default=None, help="optional -ngl value")
    c.add_argument("--verbose", action="store_true", help="stream llama-cli's stderr log live")
    c.add_argument("--list-devices", action="store_true", help="show devices llama-cli sees with --rpc, then exit")
    return p


def menu():
    print(f"{BAR}\n{'LOCALSWARM':^60}\n{'P2P Distributed Inference CLI':^60}\n{BAR}\n")
    print("[1] Inference Host / Worker\n[2] Prompting Host / Controller\n[3] Exit\n")
    try:
        choice = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return {"1": "worker", "2": "controller"}.get(choice)


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = args.mode or menu()
    if mode is None:
        return 0
    if not args.mode:  # came from menu: fill defaults
        args = parser.parse_args([mode])
    return run_worker(args) if mode == "worker" else run_controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
