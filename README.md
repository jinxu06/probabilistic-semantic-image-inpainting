# Probabilistic Semantic Image Inpainting

## How to use:

### Runing training:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0,1 --mode train
```
### Runing eval:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0 --mode test
```

### Runing inpainting:
```
python bidirectional_pixel_cnn.py configs/bidirectional_pixel_cnn_mnist.json --gpus 0 --mode inpainting
```
