# LeNet5 on FashionMNIST with Regularization Techniques

This implementation compares different regularization techniques applied to LeNet5 on the FashionMNIST dataset.

## Training the Models

To train all models, run the complete code in Google Colab. The code will:
- Train a plain LeNet5 model without regularization
- Train LeNet5 with Dropout (rate=0.5) applied to fully connected layers
- Train LeNet5 with Weight Decay (L2 regularization, rate=0.001)
- Train LeNet5 with Batch Normalization applied to convolutional and fully connected layers

Each model is trained for 20 epochs using the Adam optimizer with a learning rate of 0.001.

## Testing with Saved Weights

The code saves model weights after training. To test using saved weights:

```python
# For plain LeNet5
test_with_saved_weights('lenet5_plain.pth', LeNet5, device, test_loader)

# For LeNet5 with Dropout
test_with_saved_weights('lenet5_dropout.pth', LeNet5Dropout, device, test_loader)

# For LeNet5 with Weight Decay
test_with_saved_weights('lenet5_weight_decay.pth', LeNet5, device, test_loader)

# For LeNet5 with Batch Normalization
test_with_saved_weights('lenet5_batch_norm.pth', LeNet5BatchNorm, device, test_loader)

## Results and Conclusions

Based on the experimental results:

- All regularization techniques achieved test accuracies above 89.5%, with Weight Decay achieving the highest at 90.70%.

- Batch Normalization demonstrated the fastest initial learning but led to the largest gap between training (97.69%) and test accuracy (90.25%), indicating potential overfitting.

- Dropout showed the best generalization with the smallest training-test gap (2.41%), confirming its effectiveness against overfitting while maintaining good test performance (89.63%).

- Weight Decay provided the best balance between training performance (93.61%) and generalization (90.70%), with a moderate gap of 2.91%.

- Despite their theoretical differences, all techniques reached similar test accuracy, suggesting that for this particular dataset and architecture, the choice of regularization primarily affects training dynamics rather than ultimate generalization capability.
