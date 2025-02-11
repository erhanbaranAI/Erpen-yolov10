import os
import subprocess

# Sanal ortamı aktifleştir
venv_path = "yolov10_env"
activate_script = os.path.join(venv_path, "Scripts", "activate.bat" if os.name == "nt" else "bin/activate")
subprocess.call(f'source {activate_script}', shell=True)

# Gerekli paketleri yükle
subprocess.call("pip install --upgrade pip", shell=True)
subprocess.call("pip install git+https://github.com/THU-MIG/yolov10.git", shell=True)
subprocess.call("pip install supervision roboflow", shell=True)

# Eğitim verilerini ve önceden eğitilmiş ağırlıkları indir
home = os.getcwd()
weights_dir = os.path.join(home, "weights")
os.makedirs(weights_dir, exist_ok=True)

pretrained_weights = ["yolov10x.pt", "yolov10l.pt"]

base_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/"
for weight in pretrained_weights:
    weight_url = os.path.join(base_url, weight)
    subprocess.call(f"wget -P {weights_dir} -q {weight_url}", shell=True)

# Roboflow API anahtarı
ROBOFLOW_API_KEY = "YOUR_ROBOFLOW_API_KEY"

from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("selencakmak").project("tumor-dj2a1")
version = project.version(1)
dataset = version.download("yolov8")

# Eğitim
data_yaml_path = os.path.join(dataset.location, "data.yaml")

# YOLOv10x Modeli ile Eğitim
model_path_x = os.path.join(weights_dir, "yolov10x.pt")
subprocess.call(f"yolo task=detect mode=train epochs=10 batch=32 plots=True model={model_path_x} data={data_yaml_path}", shell=True)

# YOLOv10l Modeli ile Eğitim
model_path_l = os.path.join(weights_dir, "yolov10l.pt")
subprocess.call(f"yolo task=detect mode=train epochs=10 batch=32 plots=True model={model_path_l} data={data_yaml_path}", shell=True)
