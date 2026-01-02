# Hardware Monitor Module

AI-powered hardware health monitoring and prediction for Node.js projects.

## Installation

1. Copy this folder into your project.
2. Install Node.js dependencies:
   ```bash
   npm install onnxruntime-node
   ```
3. Ensure Python is installed with the following requirements:
   ```bash
   pip install psutil
   ```
   *(Note: `smartmontools` should be installed on the system for SMART data collection)*

## Usage

```javascript
const HardwareMonitor = require('./hardware-monitor/monitor');

async function checkHardware() {
    const monitor = new HardwareMonitor();
    await monitor.init();
    
    const results = await monitor.runFullCheck();
    console.log(results);
}

checkHardware();
```

## Folder Structure

- `monitor.js`: Node.js wrapper and prediction logic.
- `hardware_monitor.py`: Python script for raw data collection.
- `health_model.onnx`: AI model for anomaly detection.
- `health_history.json`: Local storage for tracking attribute changes.
