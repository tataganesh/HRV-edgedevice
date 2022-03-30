#Conversion
echo "Converting"
python torch_to_onnx.py --pytorch_model_path $1 
python onnx_to_tflite.py

#Testing
echo "Testing"
python run_tflite_model.py
