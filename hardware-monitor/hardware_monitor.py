import subprocess
import json
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
import platform
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_command(command):
    """Runs a shell command and returns the output."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            logging.debug(f"Command failed: {command}\nError: {result.stderr.strip()}")
            return None
        return result.stdout.strip()
    except Exception as e:
        logging.error(f"Error running command '{command}': {e}")
        return None


def get_disk_list_wmic():
    """Get list of disks using WMIC (Windows)."""
    output = run_command('wmic diskdrive get DeviceID,Caption,Status')
    disks = []
    if output:
        lines = output.splitlines()
        # Skip header
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 2:
                # Basic parsing, might need refinement based on exact output
                # Usually DeviceID is \\.\PHYSICALDRIVEX
                device_id = next((p for p in parts if "PHYSICALDRIVE" in p), None)
                if device_id:
                    disks.append(device_id)
    return disks

def get_smart_scan():
    """Scans for devices using smartctl."""
    output = run_command('smartctl --scan -j')
    if output:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None
    return None

def get_smart_attributes_smartctl(device_name, device_type=None):
    """Fetch all attributes for a specific device."""
    cmd = f'smartctl -a -j {device_name}'
    if device_type:
        cmd += f' -d {device_type}'
    
    output = run_command(cmd)
    if output:
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None
    return None

def get_wmic_status():
    """Fallback: Get simple status from WMIC."""
    output = run_command('wmic diskdrive get DeviceID,Caption,Status') 
    
    data = []
    if output:
        lines = [l.strip() for l in output.splitlines() if l.strip()]
        if len(lines) > 1:
            header = lines[0]
            for line in lines[1:]:
                data.append({"raw": line})
    return data

def get_system_metrics_fallback():
    """Get approximate system metrics using WMIC/Powershell on Windows."""
    metrics = {"cpu_percent": 0.0, "memory_percent": 0.0}
    
    # CPU Load via wmic
    try:
        cpu_out = run_command("wmic cpu get loadpercentage")
        if cpu_out:
            lines = cpu_out.splitlines()
            if len(lines) > 1:
                metrics["cpu_percent"] = float(lines[1].strip())
    except:
        pass

    # Memory via wmic
    try:
        mem_out = run_command("wmic OS get FreePhysicalMemory,TotalVisibleMemorySize /Value")
        if mem_out:
            lines = mem_out.splitlines()
            total = 0
            free = 0
            for line in lines:
                if "TotalVisibleMemorySize" in line:
                    total = float(line.split('=')[1])
                if "FreePhysicalMemory" in line:
                    free = float(line.split('=')[1])
            if total > 0:
                metrics["memory_percent"] = round(((total - free) / total) * 100, 1)
    except:
        pass
        
    return metrics

def collect_system_metrics():
    """Collects CPU, RAM, and Disk I/O usage."""
    metrics = {
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "disk_io": {}
    }
    
    if PSUTIL_AVAILABLE:
        metrics["cpu_percent"] = psutil.cpu_percent(interval=1)
        metrics["memory_percent"] = psutil.virtual_memory().percent
        
        # Disk I/O
        try:
            io_counters = psutil.disk_io_counters(perdisk=True)
            for disk, counters in io_counters.items():
                metrics["disk_io"][disk] = {
                    "read_bytes": counters.read_bytes,
                    "write_bytes": counters.write_bytes,
                    "busy_time": getattr(counters, "busy_time", 0) # Only available on Linux
                }
        except:
            pass
        return metrics
    else:
        fallback = get_system_metrics_fallback()
        metrics.update(fallback)
        return metrics

def main():
    logging.info("Starting Hardware Monitor...")
    
    report = {
        "system": platform.system(),
        "metrics": collect_system_metrics(),
        "disks": []
    }

    # 1. Try smartctl first
    scan_result = get_smart_scan()
    smart_available = scan_result is not None
    
    if smart_available and 'devices' in scan_result:
        logging.info(f"Found {len(scan_result['devices'])} devices via smartctl.")
        for dev in scan_result['devices']:
            name = dev.get('name')
            dtype = dev.get('type')
            if name:
                smart_data = get_smart_attributes_smartctl(name, dtype)
                if smart_data:
                    # Normalize essential data
                    disk_info = {
                        "device": name,
                        "model": smart_data.get("model_name", "Unknown"),
                        "smart_status": smart_data.get("smart_status", {}).get("passed"),
                        "temperature": smart_data.get("temperature", {}).get("current"),
                        "smart_attributes": smart_data.get("ata_smart_attributes", {}).get("table", []),
                        # NVMe stores attributes differently
                        "nvme_attributes": smart_data.get("nvme_smart_health_information_log", {})
                    }
                    report["disks"].append(disk_info)
    
    else:
        logging.warning("smartctl not found or failed. Falling back to WMIC.")
        wmic_data = get_wmic_status()
        report["wmic_fallback"] = wmic_data
        
        # Parse WMIC roughly
        for entry in wmic_data:
             report["disks"].append({
                 "raw_wmic": entry["raw"],
                 "status": "Unknown (WMIC Fallback)"
             })

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
