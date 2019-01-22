# Probabilistic Semantic Image Inpainting

## How to use:

### Run training:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0,1 --mode train
```
### Run eval:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0 --mode test
```

### Run inpainting:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0 --mode inpainting
```
