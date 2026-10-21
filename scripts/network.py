"""IP validation and TCP reachability."""
import errno
import ipaddress
import socket

from .config import TCP_TIMEOUT_S


def validate_ip(text: str):
    """Return (ip_string, None) or (None, error)."""
    text = text.strip()
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return None, f"'{text}' is not a valid IP address (example: 192.168.137.160)."
    if ip.is_unspecified or ip.is_multicast:
        return None, f"{ip} cannot be used as a worker address."
    return str(ip), None


def rpc_endpoint(host: str, port: int) -> str:
    """Format an RPC endpoint, adding brackets around IPv6 literals."""
    ip = ipaddress.ip_address(host)
    return f"[{host}]:{port}" if ip.version == 6 else f"{host}:{port}"


def tcp_check(host: str, port: int, timeout: float = TCP_TIMEOUT_S):
    """Return (ok, explanation). Only proves the port accepts TCP connections."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "TCP connection succeeded"
    except socket.timeout:
        return False, (f"Timed out connecting to {host}:{port}. The worker is offline, the IP is wrong, "
                       f"or a firewall is silently dropping port {port}.")
    except ConnectionRefusedError:
        return False, (f"Connection refused by {host}:{port}. The host is up but nothing is listening: "
                       f"the RPC server is not running (or uses another port).")
    except OSError as e:
        if e.errno in (errno.ENETUNREACH, errno.EHOSTUNREACH) or getattr(e, "winerror", 0) in (10051, 10065):
            return False, f"No route to {host}. Check that both PCs are on the same network/hotspot."
        return False, (f"Could not connect to {host}:{port} ({e}). Check the worker is running and that "
                       f"the firewall allows inbound TCP {port}.")
