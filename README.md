# Rain Removal Neural Network
This project involves the development and training of a neural network model for removing rain from images. The model leverages PyTorch and is designed to enhance image clarity by reducing rain effects, while maintaining the overall image structure and detail.

## Project Overview
The model architecture primarily consists of a convolutional neural network (CNN) with guided filtering to separate the rain from the underlying image structure. This is achieved by dividing the image into base and detail layers, followed by residual learning to enhance the final output.

### Key Features
- **Guided Filter: Helps in separating the rain streaks from the main image content.
- **Residual Learning: Allows the model to better focus on refining the details after initial filtering.
- **Contrast and Sharpness Enhancement: Post-processing steps to improve the visual quality of the rain-removed images.

## Installation
To run the project, ensure you have Python 3.x installed along with the required dependencies. You can install the necessary packages using:
```bash
pip install -r requirements.txt
```

## Dataset
[Dataset download (Google cloud)](https://drive.google.com/file/d/1VJWfuM30LCE5LditiexzVhyASi70xyS4/view)
##### Structure

## Training
The training script trains the model on the provided dataset. You can adjust the learning rate, number of epochs, batch size, and other hyperparameters directly in the script.

Note: The code is configured to automatically detect the available device (mps, cuda, or cpu).

To train the model, run:
```bash
python train.py
```
### Hyperparameters
* Learning Rate: 1e-4
* Features: 32
* Epochs: 0 (Adjust this as needed)
* Batch Size: 4
* Contrast Factor: 1.4
* Sharpness Factor: 1.8

## Model
The model architecture includes 16 residual blocks for learning complex patterns and a final guided filter step to refine the output. The trained model is saved as final_model_continue_residual.pth.

### Loading Pre-trained Model
If a pre-trained model exists, the script will load it automatically:
```python
checkpoint = torch.load(model_path, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## Evaluation
After training, the model is evaluated on the validation dataset. The script computes the mean squared error (MSE) between the predicted and target images as the loss metric.

## Output
Processed images are saved in the **'Train_IEEE_output_train_3'** directory with filenames indicating the batch and image number.

### 

## References
This work was inspired by and built upon the following papers:
1. **Removing rain from single images via a deep detail network** - Xueyang Fu Jiabin Huang Delu Zeng Yue Huang Xinghao Ding John Paisley IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017

## License
This project is licensed under the MIT License - see the LICENSE file for details.