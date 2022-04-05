import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Tensorflow to tflite')
parser.add_argument('--tf_model_path', help="Path to pytorch model", required=True)

args = parser.parse_args()

saved_model_dir = args.tf_model_path
tflite_model_path = 'model.tflite'

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the model
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)