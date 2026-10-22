# LocalSwarm

LocalSwarm is a Python CLI and browser dashboard for coordinating `llama.cpp` RPC inference across a controller and one or more worker laptops. The controller keeps the model file and prompt; workers expose an RPC device. An NVIDIA GPU is selected on workers when available, with CPU/system RAM as the fallback.

## Requirements

- Python 3.8 or newer
- Compatible `llama.cpp` builds with RPC support on the controller and each worker
- A trusted private network between machines
- The GGUF model file on the controller only

## Install from a source checkout

On each machine, extract or clone this project, then install it in editable mode:

```powershell
py -m pip install -e .
```

This installs the `localswarm` command. You can also invoke it as `py -m localswarm` after installation. No third-party Python runtime packages are required; `llama.cpp` executables are installed separately.

If `py` is unavailable, use `python -m pip install -e .`.

## Run a worker

Start this on each worker laptop. The worker does not need a copy of the model:

```powershell
localswarm worker --port 50052
```

## Run inference from the controller

Pass the model path that exists on the controller. List one or more worker IP addresses:

```powershell
localswarm controller --workers "192.168.1.42,192.168.1.43" --model "C:\models\qwen-4b.gguf" --prompt "Hello"
```

Set explicit percentages in this order: controller, worker 1, worker 2, and so on. Values must total 100:

```powershell
localswarm controller --workers "192.168.1.42,192.168.1.43" --split 10,60,30 --model "C:\models\qwen-4b.gguf" --prompt "Hello"
```

A single worker works as well. For example, `--split 40,60` assigns approximately 40% of layers to the controller and 60% to that worker. Layer percentages are approximate because layer sizes and compute costs vary. llama.cpp uses RPC devices and its layer offload controls to place work. For multiple worker devices, LocalSwarm also passes `--tensor-split` to apportion remote device memory; this is a llama.cpp device-placement option, not pipeline parallel scheduling.

The default split sends approximately 50% of layers to the worker pool. Use `--split-percent 60` to change the total worker share, or `--gpu-layers N` to set the remote layer count directly.

## Browser dashboard

Run the dashboard on the controller and open `http://127.0.0.1:8765`:

```powershell
localswarm ui
```

On each worker, start its dashboard and read-only telemetry endpoint:

```powershell
localswarm ui --agent-host 0.0.0.0
```

Choose **Controller** or **Worker** in the browser to configure the pool, edit layer shares, start worker RPC, submit prompts, and view hardware meters. Permit TCP ports 50052 and 8766 on each worker's private network firewall if needed. The telemetry endpoint has no authentication; use it only on a trusted LAN. The dashboard control interface should remain bound to localhost unless you specifically need remote browser access.

## Build distributable packages

From the project root, install the build frontend and create wheel and source distributions:

```powershell
py -m pip install build
py -m build
```

To create the portable source ZIP without build tools, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\package-source.ps1
```

Artifacts are written to `dist/`. The wheel includes the dashboard HTML. For a direct source deployment, share the project folder or source archive, install with `py -m pip install .`, and install compatible `llama.cpp` executables separately. The checked-in source and packaging metadata do not bundle model weights or llama.cpp binaries.
## Configuration and help

```powershell
localswarm --help
localswarm worker --help
localswarm controller --help
localswarm ui --help
```

Set `LOCALSWARM_LLAMA_CLI` or `LOCALSWARM_RPC_SERVER` to select executable paths explicitly. RPC has no authentication; never expose it to the public internet.



