# Deep Learning with PyTorch

Deep learning is widely used in applications such as self-driving cars, voice assistants, and search engines. This document explains key deep learning concepts using PyTorch with relevant code examples.

## 1. Introduction to PyTorch, a Deep Learning Library

Before building complex deep learning models, it's important to understand PyTorch, a popular deep learning framework. PyTorch provides flexible tools for building and training neural networks efficiently.

### PyTorch Basics

PyTorch operates using tensors, which are multi-dimensional arrays similar to NumPy arrays but optimized for GPU acceleration.

```python
import torch

# Creating a tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(x)
```

**Explanation:**
- `torch.tensor()` is used to create tensors, the basic data structure in PyTorch.
- Tensors enable operations such as addition, multiplication, and gradients for deep learning.

### Linear Layers

A linear layer (fully connected layer) performs a weighted sum of inputs followed by an optional activation function.

```python
import torch.nn as nn

# Defining a simple linear model
model = nn.Linear(in_features=2, out_features=1)
print(model)
```

**Explanation:**
- `nn.Linear()` creates a layer with `in_features` as input neurons and `out_features` as output neurons.
- This layer applies a linear transformation: `output = weight * input + bias`.

## 2. Neural Network Architecture and Hyperparameters

To train a neural network, you need to understand activation functions, loss functions, and gradients.

### Activation Functions

Activation functions introduce non-linearity, helping networks learn complex patterns.

```python
import torch.nn.functional as F

# Example of ReLU activation function
def relu_activation(x):
    return F.relu(x)
```

**Explanation:**
- The ReLU (Rectified Linear Unit) activation function sets negative values to zero while keeping positive values unchanged.
- This prevents issues like the vanishing gradient problem in deep networks.

### Loss Function and Gradients

Loss functions measure how far the model's predictions are from actual values. Gradients help update model parameters during training.

```python
# Mean Squared Error Loss
loss_fn = nn.MSELoss()

# Example tensors
y_pred = torch.tensor([2.5, 3.0])
y_true = torch.tensor([3.0, 3.0])

loss = loss_fn(y_pred, y_true)
print("Loss:", loss.item())
```

**Explanation:**
- `MSELoss()` computes the Mean Squared Error, commonly used for regression tasks.
- `loss.backward()` (not shown here) computes gradients used for optimization.

## 3. Training a Neural Network with PyTorch

After understanding key components, you can train a simple neural network using an optimizer and a training loop.

### Training Loop

The training loop involves forward propagation, loss computation, backpropagation, and parameter updates.

```python
# Sample training loop
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(torch.randn(1, 2))
    loss = loss_fn(output, torch.tensor([1.0]))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

**Explanation:**
- `optimizer.zero_grad()` clears previous gradients.
- `output = model(input)` performs a forward pass.
- `loss.backward()` computes gradients for backpropagation.
- `optimizer.step()` updates model parameters using gradients.

## 4. Evaluating and Improving Models

Tracking key metrics during training ensures the model is learning correctly and generalizing well.

### Calculating Accuracy

Accuracy is a common metric for classification tasks.

```python
def accuracy(predictions, labels):
    correct = (predictions.argmax(dim=1) == labels).sum().item()
    return correct / len(labels)
```

**Explanation:**
- `argmax(dim=1)` finds the predicted class with the highest probability.
- The function compares predictions with actual labels and calculates the accuracy.

### Avoiding Overfitting

Overfitting occurs when a model memorizes training data instead of generalizing. Dropout helps prevent this by randomly deactivating neurons during training.

```python
# Applying dropout
model_with_dropout = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(10, 1)
)
```

**Explanation:**
- `Dropout(p=0.5)` randomly disables 50% of neurons during training to encourage generalization.
- Regularization techniques like dropout and weight decay help prevent overfitting.

---

This document provides a foundational understanding of deep learning with PyTorch. You can expand it by experimenting with different models and techniques. Happy coding!

