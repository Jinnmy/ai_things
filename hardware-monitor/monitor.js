const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');

class HardwareMonitor {
    constructor(options = {}) {
        const baseDir = __dirname;
        this.modelPath = path.resolve(options.modelPath || path.join(baseDir, 'health_model.onnx'));
        this.historyPath = path.resolve(options.historyPath || path.join(baseDir, 'health_history.json'));
        this.pythonPath = options.pythonPath || 'python';
        this.scriptPath = path.join(baseDir, 'hardware_monitor.py');
        this.history = this._loadHistory();
        this.session = null;
    }

    async init() {
        if (!fs.existsSync(this.modelPath)) {
            throw new Error(`Model file not found at ${this.modelPath}`);
        }
        this.session = await ort.InferenceSession.create(this.modelPath);
    }

    _loadHistory() {
        if (fs.existsSync(this.historyPath)) {
            try {
                return JSON.parse(fs.readFileSync(this.historyPath, 'utf8'));
            } catch (e) {
                return {};
            }
        }
        return {};
    }

    _saveHistory() {
        fs.writeFileSync(this.historyPath, JSON.stringify(this.history, null, 2));
    }

    normalizeData(smartTable, nvmeData, systemMetrics, deviceId = "unknown") {
        const targetIds = [5, 9, 10, 187, 194, 197, 198, 199];
        let vector = [];

        const getRaw = (id) => {
            const row = (smartTable || []).find(r => r.id === id);
            return row ? (row.raw?.value || 0) : 0;
        };

        let currentVals = {};
        targetIds.forEach(tid => {
            const val = getRaw(tid);
            currentVals[tid] = val;
            let norm = 0;
            if (tid === 194) norm = Math.min(Math.max(val, 0), 100) / 100.0;
            else if (tid === 9) norm = Math.min(val, 43800) / 43800.0;
            else norm = val > 0 ? 1.0 : 0.0;
            vector.push(norm);
        });

        let percentageUsed = (nvmeData?.percentage_used || 0) / 100.0;
        let criticalWarning = (nvmeData?.critical_warning || 0) > 0 ? 1.0 : 0.0;
        let mediaErrors = (nvmeData?.media_errors || 0) > 0 ? 1.0 : 0.0;
        vector.push(percentageUsed, criticalWarning, mediaErrors);

        let busyPct = 0;
        const diskIo = systemMetrics?.disk_io || {};
        const firstDisk = Object.values(diskIo)[0];
        if (firstDisk) {
            busyPct = Math.min((firstDisk.busy_time || 0) / 1000.0, 1.0);
        }
        vector.push(busyPct);

        let deltaRealloc = 0;
        let deltaPending = 0;
        if (this.history[deviceId] && this.history[deviceId].length > 0) {
            const prev = this.history[deviceId][this.history[deviceId].length - 1];
            deltaRealloc = currentVals[5] > (prev["5"] || 0) ? 1.0 : 0.0;
            deltaPending = currentVals[197] > (prev["197"] || 0) ? 1.0 : 0.0;
        }
        vector.push(deltaRealloc, deltaPending);

        if (deviceId !== "unknown") {
            const entry = { ...currentVals, timestamp: Date.now() };
            if (!this.history[deviceId]) this.history[deviceId] = [];
            this.history[deviceId].push(entry);
            this.history[deviceId] = this.history[deviceId].slice(-10);
            this._saveHistory();
        }

        return new Float32Array(vector);
    }

    async getReport() {
        try {
            const output = execSync(`${this.pythonPath} "${this.scriptPath}"`, { encoding: 'utf8' });
            return JSON.parse(output);
        } catch (err) {
            throw new Error(`Failed to run hardware monitor script: ${err.message}`);
        }
    }

    async predict(smartTable, nvmeData, systemMetrics, deviceId = "unknown") {
        if (!this.session) await this.init();

        const inputData = this.normalizeData(smartTable, nvmeData, systemMetrics, deviceId);
        const tensor = new ort.Tensor('float32', inputData, [1, 14]);
        const results = await this.session.run({ input: tensor });
        const output = results.output.data;

        let mse = 0;
        for (let i = 0; i < inputData.length; i++) {
            mse += Math.pow(inputData[i] - output[i], 2);
        }
        mse /= inputData.length;

        const threshold = 0.05;
        return {
            anomaly_score: parseFloat(mse.toFixed(4)),
            status: mse > threshold ? "Warning: Anomaly Detected" : "Healthy",
            input_vector: Array.from(inputData)
        };
    }

    async runFullCheck() {
        const report = await this.getReport();
        const results = [];

        for (const disk of report.disks) {
            const prediction = await this.predict(
                disk.smart_attributes,
                disk.nvme_attributes,
                report.metrics,
                disk.device || "unknown_disk"
            );
            results.push({
                device: disk.device,
                model: disk.model,
                prediction
            });
        }

        return {
            timestamp: new Date().toISOString(),
            metrics: report.metrics,
            disks: results
        };
    }
}

module.exports = HardwareMonitor;

if (require.main === module) {
    (async () => {
        const monitor = new HardwareMonitor();
        try {
            console.log("Running hardware check...");
            const results = await monitor.runFullCheck();
            console.log(JSON.stringify(results, null, 2));
        } catch (err) {
            console.error(err);
        }
    })();
}
