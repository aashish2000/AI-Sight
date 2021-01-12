# Install Package: pip install tflite_support_nightly

# Change path of tflite model, labelmap file and save location
from tflite_support.metadata_writers import object_detector

ObjectDetectorWriter = object_detector.MetadataWriter
_MODEL_PATH = "/content/ssd_mobilenet_v3_small/model.tflite"
_LABEL_FILE = "/content/ssd_mobilenet_v3_small/labelmap.txt"
_SAVE_TO_PATH = "mobilenet_v3_small_metadata.tflite"

with open(_MODEL_PATH, "rb") as file:
  model_buffer = file.read()

writer = ObjectDetectorWriter.create_for_inference(
    model_buffer, [128], [128], [_LABEL_FILE])
new_model = writer.populate()

with open(_SAVE_TO_PATH, "wb") as file:
  file.write(new_model)
