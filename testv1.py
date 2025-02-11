import cv2
import os
import supervision as sv
from ultralytics import YOLOv10

# Eğitilmiş modelinizi yükleyin
model_path = 'best.pt'  # Eğitilmiş modelinizin yolunu buraya yazın
model = YOLOv10(model_path)

# Video dosyasını açın
video_path = "video.mp4"  # Video dosyanızın yolunu buraya yazın
cap = cv2.VideoCapture(video_path)

# Video yazıcıyı ayarlayın
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Geçici dosya dizinini oluşturun
temp_dir = "temp_frames"
os.makedirs(temp_dir, exist_ok=True)
frame_index = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(source=frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

    # İşlenmiş kareyi geçici dosya olarak kaydedin
    frame_file = os.path.join(temp_dir, f"frame_{frame_index:04d}.png")
    cv2.imwrite(frame_file, annotated_frame)
    frame_index += 1

    out.write(annotated_frame)

cap.release()
out.release()

# Geçici dosya dizinini temizleyin
for file_name in os.listdir(temp_dir):
    file_path = os.path.join(temp_dir, file_name)
    if os.path.isfile(file_path):
        os.unlink(file_path)
os.rmdir(temp_dir)
