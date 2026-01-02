import torch
import torch.nn as nn
import torch.optim as optim
import json
import logging
import os
import numpy as np
import download_models


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),  # Latent space
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid() # Attributes normalized 0-1
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

class HealthPredictor:
    def __init__(self, model_path='health_model.pth', history_path='health_history.json'):
        self.model_path = model_path
        self.history_path = history_path
        # Expanded attributes:
        # 0-7: ATA SMART (5, 9, 10, 187, 194, 197, 198, 199)
        # 8: NVMe Percentage Used
        # 9: NVMe Critical Warning
        # 10: NVMe Media Errors
        # 11: System Disk Busy % (mocked or from psutil)
        # 12: Delta Reallocated (ID 5)
        # 13: Delta Pending (ID 197)
        self.input_dim = 14 
        self.model = Autoencoder(self.input_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        self.history = self._load_history()
        
        if os.path.exists(self.model_path):
            logging.info(f"Loading existing model from {self.model_path}")
            try:
                self.model.load_state_dict(torch.load(self.model_path))
            except:
                logging.warning("Model architecture mismatch. Re-initializing.")
            self.model.eval()
        else:
            logging.info(f"Model {self.model_path} not found locally. Attempting to download...")
            if download_models.check_and_download_models():
                try:
                    self.model.load_state_dict(torch.load(self.model_path))
                    self.model.eval()
                    logging.info("Model downloaded and loaded successfully.")
                except Exception as e:
                    logging.error(f"Failed to load downloaded model: {e}")
            else:
                logging.info("Auto-download failed. Initialized new model.")


    def _load_history(self):
        if os.path.exists(self.history_path):
            with open(self.history_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_history(self):
        with open(self.history_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def normalize_data(self, smart_table, nvme_data=None, system_metrics=None, device_id="unknown"):
        """
        Extracts and normalizes data into a 14-dim tensor.
        """
        target_ids = [5, 9, 10, 187, 194, 197, 198, 199]
        vector = []
        
        # 1. ATA SMART Attributes (0-7)
        def get_raw(id_):
            for row in smart_table:
                if row.get('id') == id_:
                    return row.get('raw', {}).get('value', 0)
            return 0
            
        current_vals = {}
        for tid in target_ids:
            val = get_raw(tid)
            current_vals[tid] = val
            if tid == 194: # Temp
                norm = min(max(val, 0), 100) / 100.0
            elif tid == 9: # Power Hours
                norm = min(val, 43800) / 43800.0
            else:
                norm = 1.0 if val > 0 else 0.0 
            vector.append(norm)

        # 2. NVMe Specifics (8-10)
        percentage_used = 0
        critical_warning = 0
        media_errors = 0
        if nvme_data:
            percentage_used = nvme_data.get("percentage_used", 0) / 100.0
            critical_warning = 1.0 if nvme_data.get("critical_warning", 0) > 0 else 0.0
            media_errors = 1.0 if nvme_data.get("media_errors", 0) > 0 else 0.0
        
        vector.extend([percentage_used, critical_warning, media_errors])

        # 3. System I/O Busy % (11)
        busy_pct = 0
        if system_metrics:
            # Aggregate or pick disk-specific if possible
            disk_io = system_metrics.get("disk_io", {})
            # For demo simplified: just use a high-level heuristic or first disk found
            if disk_io:
                first_disk = list(disk_io.values())[0]
                # Busy time is usually in ms, but relative to time passed. 
                # Here we just normalize a high value.
                busy_pct = min(first_disk.get("busy_time", 0) / 1000.0, 1.0)
        vector.append(busy_pct)

        # 4. Historical Deltas (12-13)
        delta_realloc = 0
        delta_pending = 0
        if device_id in self.history:
            prev = self.history[device_id][-1] # Last record
            delta_realloc = 1.0 if current_vals[5] > prev.get("5", 0) else 0.0
            delta_pending = 1.0 if current_vals[197] > prev.get("197", 0) else 0.0
        
        vector.extend([delta_realloc, delta_pending])

        # Update History
        if device_id != "unknown":
            entry = {str(k): v for k, v in current_vals.items()}
            entry["timestamp"] = os.times().elapsed # Simplified timestamp
            if device_id not in self.history:
                self.history[device_id] = []
            self.history[device_id].append(entry)
            self.history[device_id] = self.history[device_id][-10:] # Keep last 10
            self._save_history()

        return torch.tensor([vector], dtype=torch.float32)

    def train_calibration(self, data_samples, epochs=100):
        """
        Train the model on 'data_samples'.
        Assume data_samples is a list of normalized tensors (or raw tables to be normalized).
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for sample in data_samples:
                # If sample is raw smart table (demo fallback)
                if isinstance(sample, list):
                    tensor = self.normalize_data(sample)
                else:
                    tensor = sample
                
                self.optimizer.zero_grad()
                output = self.model(tensor)
                loss = self.criterion(output, tensor)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logging.debug(f"Epoch {epoch}, Loss: {total_loss}")
        
        torch.save(self.model.state_dict(), self.model_path)
        logging.info("Model calibrated and saved.")

    def predict(self, smart_table, nvme_data=None, system_metrics=None, device_id="unknown"):
        """
        Returns anomaly score and prediction.
        """
        self.model.eval()
        with torch.no_grad():
            inp = self.normalize_data(smart_table, nvme_data, system_metrics, device_id)
            reconstruction = self.model(inp)
            loss = self.criterion(reconstruction, inp).item()
            
            # Heuristic threshold for anomaly
            threshold = 0.05 
            
            status = "Healthy"
            if loss > threshold:
                status = "Warning: Anomaly Detected"
                
            return {
                "anomaly_score": round(loss, 4),
                "status": status,
                "input_vector": inp.tolist(),
                "reconstruction": reconstruction.tolist()
            }

if __name__ == "__main__":
    # Test Dummy
    predictor = HealthPredictor(model_path='health_model_v2.pth')
    
    # 1. Simulate Healthy Drive Data
    healthy_table = [
        {'id': 5, 'raw': {'value': 0}},
        {'id': 9, 'raw': {'value': 1000}},
        {'id': 10, 'raw': {'value': 0}},
        {'id': 187, 'raw': {'value': 0}},
        {'id': 194, 'raw': {'value': 35}},
        {'id': 197, 'raw': {'value': 0}},
        {'id': 198, 'raw': {'value': 0}},
        {'id': 199, 'raw': {'value': 0}},
    ]
    
    # Simulate normal system metrics
    sys_ok = {"disk_io": {"disk0": {"busy_time": 50}}}
    
    # Force a few history entries
    for _ in range(3):
        predictor.normalize_data(healthy_table, system_metrics=sys_ok, device_id="demo_ssd")

    logging.info("Training on healthy data...")
    # Generate 10 healthy tensors for training
    tensors = [predictor.normalize_data(healthy_table, system_metrics=sys_ok, device_id="demo_ssd") for _ in range(10)]
    predictor.train_calibration(tensors)
    
    # 2. Predict Healthy
    res = predictor.predict(healthy_table, system_metrics=sys_ok, device_id="demo_ssd")
    print(f"Healthy Test: {res['status']} (Score: {res['anomaly_score']})")
    
    # 3. Predict Failing (Delta Reallocated)
    failing_table = list(healthy_table)
    # Reallocated count increases from 0 to 10
    failing_table[0] = {'id': 5, 'raw': {'value': 10}} 
    
    res_fail = predictor.predict(failing_table, device_id="demo_ssd")
    print(f"Failing Test (Delta Error): {res_fail['status']} (Score: {res_fail['anomaly_score']})")
    
    # 4. Predict NVMe Life Warning
    res_nvme = predictor.predict(healthy_table, nvme_data={"percentage_used": 99}, device_id="demo_nvme")
    print(f"NVMe Life Warning: {res_nvme['status']} (Score: {res_nvme['anomaly_score']})")
