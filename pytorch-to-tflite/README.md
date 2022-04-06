# Sample Conversion Example

This example will try to convert Mobilenetv2 model.

## Conversion

### Convert Upsampler pytorch model to C code for Tflite

```bash
bash convert.sh upsampler /path/to/model
```

### Convert regressor pytorch model to C code for Tflite

```bash
bash convert.sh regressor /path/to/tfmodelfolder 
```