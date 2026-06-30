# Modelos YOLOv8 (TFLite)

Esta carpeta contiene los modelos `.tflite` usados por el ejemplo
`yolo_classifier_page.dart`.

## Descargar el modelo YOLOv8s (versión S, la más pequeña)

1. Visita https://github.com/onnx/models/yolov8 o exporta tu propio modelo con
   Ultralytics:

   ```bash
   pip install ultralytics
   yolo export model=yolov8s.pt format=tflite imgsz=640
   ```

   Esto genera `yolov8s.tflite` (entrada: 1x640x640x3 float32,
   salida: 1x84x8400 float32 → [cx, cy, w, h, 80 clases]).

2. Copia el archivo a esta carpeta:

   ```bash
   cp yolov8s.tflite assets/models/yolov8s.tflite
   ```

3. Vuelve a ejecutar `flutter pub run` o recompila la app. El archivo se
   incluye en el bundle automáticamente (ver `pubspec.yaml` → `assets:`).

## Nombres esperados por el código

- `assets/models/yolov8s.tflite` — modelo cuantizado o float32.

Si el archivo no existe, el ejemplo YOLO mostrará un mensaje de error
informativo en lugar de hacer crash.