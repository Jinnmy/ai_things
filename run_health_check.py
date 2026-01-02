import json
import logging
import sys
import random
from hardware_monitor import get_smart_scan, get_smart_attributes_smartctl, get_wmic_status, collect_system_metrics
from health_predictor import HealthPredictor
import download_models


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_dummy_smart_data(healthy=True):
    """Generates dummy SMART data for demonstration purposes."""
    # Base values
    realloc = 0 if healthy else random.randint(10, 200)
    temp = random.randint(30, 45)
    hours = random.randint(1000, 20000)
    
    return [
        {'id': 5, 'raw': {'value': realloc}}, # Reallocated Sector Count
        {'id': 9, 'raw': {'value': hours}},    # Power On Hours
        {'id': 10, 'raw': {'value': 0}},       # Spin Retry Count
        {'id': 187, 'raw': {'value': 0}},      # Reported Uncorrectable Errors
        {'id': 194, 'raw': {'value': temp}},   # Temperature
        {'id': 197, 'raw': {'value': 0}},      # Current Pending Sector Count
        {'id': 198, 'raw': {'value': 0}},      # Offline Uncorrectable
        {'id': 199, 'raw': {'value': 0}},      # UDMA CRC Error Count
    ]

def main():
    print("AI Hardware Health Manager Initialized...")
    
    # Pre-flight check for models
    download_models.check_and_download_models()
    
    predictor = HealthPredictor()

    
    system_metrics = collect_system_metrics()
    scan_result = get_smart_scan()
    real_disks_found = False
    
    final_report = {
        "system_status": system_metrics,
        "drives": []
    }

    if scan_result and 'devices' in scan_result:
        for dev in scan_result['devices']:
            name = dev.get('name')
            dtype = dev.get('type')
            if name:
                logging.info(f"Analyzing Drive: {name}")
                smart_data = get_smart_attributes_smartctl(name, dtype)
                if smart_data:
                    real_disks_found = True
                    nvme_log = smart_data.get("nvme_smart_health_information_log", {})
                    # Predict
                    prediction = predictor.predict(
                        table, 
                        nvme_data=nvme_log, 
                        system_metrics=system_metrics,
                        device_id=name
                    )
                    
                    drive_report = {
                        "device": name,
                        "model": smart_data.get("model_name", "Unknown"),
                        "health_prediction": prediction,
                        "smart_data_source": "Real Hardware"
                    }
                    final_report["drives"].append(drive_report)

    if not real_disks_found:
        logging.warning("No S.M.A.R.T capable drives found (smartctl missing or VM environment).")
        logging.info("Running in DEMO MODE with simulated drives.")
        
        # Simulate Healthy Drive
        dummy_healthy = generate_dummy_smart_data(healthy=True)
        pred_good = predictor.predict(dummy_healthy)
        final_report["drives"].append({
            "device": "/dev/demo_disk_1 (Simulated)",
            "model": "Generic AI Demo SSD",
            "health_prediction": pred_good,
            "smart_data_source": "Simulated (Healthy)"
        })
        
        # Simulate Failing Drive
        dummy_bad = generate_dummy_smart_data(healthy=False)
        pred_bad = predictor.predict(dummy_bad)
        final_report["drives"].append({
            "device": "/dev/demo_disk_2 (Simulated)",
            "model": "Old Mechanical HDD",
            "health_prediction": pred_bad,
            "smart_data_source": "Simulated (Failing)"
        })

    # Output JSON for frontend/dashboard
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    main()
