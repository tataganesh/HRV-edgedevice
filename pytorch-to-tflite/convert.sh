#Conversion
if [ $1 == "upsampler" ]
then
    echo "Converting pytorch model to tflite"
    python torch_to_onnx.py --pytorch_model_path $2
    python onnx_to_tflite.py
else
    echo "Converting Tensorflow model to tflite"
    python tf_to_tflite.py --tf_model_path $2

 echo "Conversion Completed. Saved as model.tflite"
fi

#Testing 
echo "Testing"
python run_tflite_model.py


# Convert to CC
echo "Generating C Source file"
MODEL_TFLITE="model.tflite"
MODEL_TFLITE_MICRO="$1.cc"

apt-get install xxd
# Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
xxd -i $MODEL_TFLITE > $MODEL_TFLITE_MICRO
# Update variable names
REPLACE_TEXT=${MODEL_TFLITE//./_}
sed -i 's/'$REPLACE_TEXT'/g_model/g' $MODEL_TFLITE_MICRO

echo "C Source file generated. Saved as $1.cc"

# echo "C Source file"
# cat $MODEL_TFLITE_MICRO
# echo "$(cat $MODEL_TFLITE_MICRO)"