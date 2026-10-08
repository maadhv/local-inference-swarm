import os
import sys
import time
import socket
import json
import subprocess
import platform

PORT = 50052
TELEMETRY_PORT = 50053

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

def get_system_telemetry():
    """Collect basic CPU, RAM, and OS hardware specs."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "Generic CPU",
        "ram_gb": "N/A",
        "vram_info": "CUDA/Integrated Hardware"
    }
    
    # Try getting RAM via psutil if available
    try:
        import psutil
        ram = psutil.virtual_memory()
        info["ram_gb"] = f"{ram.total / (1024**3):.1f} GB (Free: {ram.available / (1024**3):.1f} GB)"
    except ImportError:
        info["ram_gb"] = "Install psutil for accurate RAM metrics"

    # Try getting NVIDIA GPU info via nvidia-smi if present
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True
        )
        name, total, free = smi_out.strip().split(',')
        info["vram_info"] = f"{name.strip()} | Total: {total.strip()} MB | Free: {free.strip()} MB"
    except Exception:
        pass

    return info

def run_telemetry_server():
    """Background socket server sending system telemetry to the Prompt Host."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', TELEMETRY_PORT))
    server.listen(1)
    
    conn, _ = server.accept()
    data = json.dumps(get_system_telemetry())
    conn.sendall(data.encode('utf-8'))
    conn.close()
    server.close()

# -------------------------------------------------------------------
# MODE 1: INFERENCE HOST (Worker PC)
# -------------------------------------------------------------------
def mode_inference_host():
    local_ip = get_local_ip()
    print("\n" + "="*60)
    print("           [MODE: INFERENCE HOST / WORKER NODE]")
    print("="*60)
    print(f"[+] Listening on Local IP : {local_ip}")
    print(f"[+] RPC Daemon Port       : {PORT}")
    print(f"[+] Telemetry Port        : {TELEMETRY_PORT}")
    print("[+] Waiting for connection from Prompting PC...\n")

    # Serve telemetry first in a lightweight handshake
    try:
        run_telemetry_server()
        print("[+] Telemetry exchanged successfully with Prompting PC.")
    except Exception as e:
        print(f"[!] Telemetry exchange skipped/failed: {e}")

    # Launch llama.cpp RPC server daemon
    cmd = ["./rpc-server", "-H", "0.0.0.0", "-p", str(PORT)]
    print(f"[+] Launching RPC Daemon: {' '.join(cmd)}")
    print("[+] Worker active. Press Ctrl+C to terminate connection.\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n[-] Session ended. Closing connection...")

# -------------------------------------------------------------------
# MODE 2: PROMPTING HOST (Client/Master PC)
# -------------------------------------------------------------------
def mode_prompting_host():
    print("\n" + "="*60)
    print("           [MODE: PROMPTING HOST / MASTER CLIENT]")
    print("="*60)
    
    worker_ip = input("Enter IP Address of Inference Host PC: ").strip()
    model_path = input("Enter Local Path to GGUF Model File  : ").strip()

    # Step 1: Fetch Telemetry from Worker
    print(f"\n[+] Connecting to {worker_ip} to retrieve hardware stats...")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(4.0)
        s.connect((worker_ip, TELEMETRY_PORT))
        raw_data = s.recv(2048).decode('utf-8')
        s.close()
        
        stats = json.loads(raw_data)
        print("\n" + "-"*50)
        print("         REMOTE WORKER HARDWARE STATS")
        print("-"*50)
        print(f" OS           : {stats['os']}")
        print(f" CPU          : {stats['cpu']}")
        print(f" RAM          : {stats['ram_gb']}")
        print(f" GPU / VRAM   : {stats['vram_info']}")
        print("-"*50 + "\n")
    except Exception as e:
        print(f"[!] Warning: Could not fetch telemetry ({e}). Proceeding with connection...")

    # Step 2: Input Prompt
    prompt = input("\nEnter Prompt: ").strip()
    
    # Step 3: Execute Distributed Inference
    remote_rpc = f"{worker_ip}:{PORT}"
    cmd = [
        "./llama-cli",
        "-m", model_path,
        "--rpc", remote_rpc,
        "-p", prompt,
        "-n", "128"  # max tokens to generate
    ]

    print("\n[+] Establishing RPC Link and initiating distributed generation...\n")
    print("="*60)
    
    start_time = time.time()
    try:
        # Stream token generation to stdout in real time
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='', flush=True)
        process.wait()
    except KeyboardInterrupt:
        print("\n[-] Inference interrupted by user.")
    
    elapsed = time.time() - start_time
    print("="*60)
    print(f"\n[+] Task Completed in {elapsed:.2f} seconds.")
    print("[+] Closing RPC Connection.")

# -------------------------------------------------------------------
# MAIN ROUTER
# -------------------------------------------------------------------
def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("===========================================================")
    print("       LocalSwarm: P2P Distributed Inference CLI           ")
    print("===========================================================")
    print("  Local IP Address:", get_local_ip())
    print("===========================================================")
    print("  [1] Act as Inference Host  (Worker Node)")
    print("  [2] Act as Prompting Host  (Client / Controller)")
    print("  [3] Exit")
    print("===========================================================")
    
    choice = input("Select PC Role (1-3): ").strip()
    
    if choice == "1":
        mode_inference_host()
    elif choice == "2":
        mode_prompting_host()
    else:
        sys.exit()

if __name__ == "__main__":
    main()