"""Defaults. Nothing machine-specific (IPs, model paths) lives here."""
import os

DEFAULT_RPC_PORT = 50052
DEFAULT_BIND = "0.0.0.0"          # LAN test only; do not expose to the internet
TCP_TIMEOUT_S = 5.0
DEFAULT_N_PREDICT = 128
LOCAL_DEVICE_DEFAULT = "none"     # controller must not touch the Intel iGPU

IS_WINDOWS = os.name == "nt"
LLAMA_CLI_NAMES = ["llama-cli.exe"] if IS_WINDOWS else ["llama-cli"]
# Newer llama.cpp builds ship `rpc-server`; the project spec uses `ggml-rpc-server`.
RPC_SERVER_NAMES = (["ggml-rpc-server.exe", "rpc-server.exe"] if IS_WINDOWS
                    else ["ggml-rpc-server", "rpc-server"])

ENV_LLAMA_CLI = "LOCALSWARM_LLAMA_CLI"
ENV_RPC_SERVER = "LOCALSWARM_RPC_SERVER"
