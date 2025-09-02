from ultralytics import YOLO

model = YOLO('yolov10n.pt') 

results = model.predict(source='0', show=True, conf=0.25, save= True)  