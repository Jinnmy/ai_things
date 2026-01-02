import os
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration - User should update the BASE_URL once they have a public repo/release
BASE_URL = "https://raw.githubusercontent.com/user/repo/main/" 

MODELS = {
    "health_model.pth": 7385,
    "health_model_v2.pth": 8271,
    "health_model.onnx": 19004,
}

def download_file(url, destination):
    """Downloads a file from a URL to a destination."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Successfully downloaded: {destination}")
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def check_and_download_models(base_url=BASE_URL, force=False):
    """Checks for missing models and downloads them."""
    missing_any = False
    for model_name, expected_size in MODELS.items():
        if not os.path.exists(model_name) or force:
            logging.info(f"Model {model_name} missing or force update. Downloading...")
            url = base_url + model_name
            if download_file(url, model_name):
                # Verify size if possible
                actual_size = os.path.getsize(model_name)
                if actual_size < 1000: # LFS pointers are small text files
                    logging.warning(f"File {model_name} seems too small ({actual_size} bytes). It might be an LFS pointer instead of the actual model data.")
                else:
                    logging.info(f"File {model_name} downloaded successfully.")
            else:
                missing_any = True
        else:
            logging.debug(f"Model {model_name} already exists.")
            
    return not missing_any

if __name__ == "__main__":
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else BASE_URL
    check_and_download_models(base_url=url)
