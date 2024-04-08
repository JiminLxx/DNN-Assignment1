# DNN-Assignment1
```python
import torch
print("Using torch", torch.__version__)

import numpy as np
print("Using numpy", np.__version__)

# To get the consistent result, we can set seed
np.random.seed(42)
torch.manual_seed(42)
```

## Task1

### Numpy

```python
# input weight
x1 = np.array([1.0, 2.0, 3.0])
x2 = np.array([4.0, 5.0, 6.0])

# NN weight
w1 = np.array([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2]])
w2 = np.array([[0.2, 0.3],
               [0.4, 0.5],
               [0.6, 0.7],
               [0.8, 0.9]])

# functions
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    t = np.exp(x)
    return t / t.sum(axis=0, keepdims=True)

# Calculate
h1 = relu(x1 @ w1)
y_pred1 = softmax(h1 @ w2)

h2 = relu(x2 @ w1)
y_pred2 = softmax(h2 @ w2)

# Print
for x, y_pred in zip([x1, x2], [y_pred1, y_pred2]):
  print(f"The output of {x} is {y_pred}")
  # The difference between target and prediction of two nodes are same.
```

### pytorch

```python
# input weight
x1 = torch.tensor([1.0, 2.0, 3.0])
x2 = torch.tensor([4.0, 5.0, 6.0])

# NN weight
w1 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8],
               [0.9, 1.0, 1.1, 1.2]])
w2 = torch.tensor([[0.2, 0.3],
               [0.4, 0.5],
               [0.6, 0.7],
               [0.8, 0.9]])

# functions
def relu(x):
    return torch.max(torch.tensor(0), x)
    # return torch.relu(x)

def softmax(x):
    t = torch.exp(x)
    return t / t.sum(axis=0, keepdims=True)

# Calculate
h1 = relu(x1 @ w1)
y_pred1 = softmax(h1 @ w2)

h2 = relu(x2 @ w1)
y_pred2 = softmax(h2 @ w2)

# Print
for x, y_pred in zip([x1, x2], [y_pred1, y_pred2]):
  print(f"The output of {x} is {y_pred}")
  # The difference between target and prediction of two nodes are same.
```


## Task2

### Numpy

```python
# Target
y1 = np.array([0, 1])
y2 = np.array([1, 0])

# [Loss function] Cross Entropy
def cross_entropy(y_pred, y):
  m = y.shape[0]
  y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9) # Make y_pred to be between 1e-9 and (1 - 1e-9)
  return -np.sum(y * np.log(y_pred))

# [Backpropagation]
def backward(x, y, h, y_pred):
  # 입력 x와 은닉층 활성화 h를 2차원 배열로 만들기
  x = x.reshape(1, -1) if x.ndim == 1 else x
  h = h.reshape(1, -1) if h.ndim == 1 else h

  dL_dy_pred = (y_pred - y).reshape(1, -1) # Gradient of Cross entropy and Softmax
  dh = (h > 0).astype(float) # Gradient of ReLU in hidden layer
  dw1 = x.T @ ((dL_dy_pred @ w2.T) * dh) # Gradinet of w1 using chain rule
  dw2 = h.T @ dL_dy_pred # h.T shape is (4, 1), dL_dy_pred shape is (1, 2)
  return dw1, dw2


for x, y, h, y_pred in zip([x1, x2], [y1, y2], [h1, h2], [y_pred1, y_pred2]):
  print(f"Loss: {cross_entropy(y_pred, y)}")
  # Gradient
  dw1, dw2 = backward(x, y, h, y_pred)
  print(f"The w1 gradient of {x} is \n {dw1}\n")
  # Each columns have same value because the difference between target and prediction of two output nodes are same.

```

### pytorch

```python
# Target
y1 = torch.tensor([0, 1])
y2 = torch.tensor([1, 0])

# [Loss function] Cross Entropy
def cross_entropy(y_pred, y):
  m = y.shape[0]
  y_pred = torch.clip(y_pred, 1e-9, 1 - 1e-9) # Make y_pred to be between 1e-9 and (1 - 1e-9)
  return -torch.sum(y * torch.log(y_pred))

# [Backpropagation]
def backward(x, y, h, y_pred):
  # Make 2D array input x and hidden activation h
  x = x.reshape(1, -1) if x.ndim == 1 else x
  h = h.reshape(1, -1) if h.ndim == 1 else h

  dL_dy_pred = (y_pred - y).reshape(1, -1) # Gradient of Cross entropy and Softmax
  dh = (h > 0).float() # Gradient of ReLU in hidden layer
  dw1 = torch.matmul( x.T, (torch.matmul(dL_dy_pred, w2.T) * dh) ) # Gradinet of w1 using chain rule
  dw2 = h.T @ dL_dy_pred # h.T shape is (4, 1), dL_dy_pred shape is (1, 2)
  return dw1.type(torch.float), dw2.type(torch.float)


for x, y, h, y_pred in zip([x1, x2], [y1, y2], [h1, h2], [y_pred1, y_pred2]):
  print(f"Loss: {cross_entropy(y_pred, y)}")
  # Gradient
  dw1, dw2 = backward(x, y, h, y_pred)
  print(f"The w1 gradient of {x} is \n {dw1}\n")
  # Each columns have same value because the difference between target and prediction of two output nodes are same.

```



## Task3

### Numpy

```python
num_epochs = 100
learning_rate = 0.01
prob = 0.4

w1_initial = w1.copy()
w2_initial = w2.copy()

# [Function] Dropout: To preventing overfitting, make some nodes' output 0 randomly in training
class Dropout:
    def __init__(self, prob = prob): # prob: the probability to remove neuron(1: remove, 0: keep)
        assert 0 <= prob <= 1 # Dropout probability must be in range [0, 1]
        self.prob = prob
        self.x1 = np.array([1.0, 2.0, 3.0])
        self.x2 = np.array([4.0, 5.0, 6.0])
        self.w1 = np.array([[0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.9, 1.0, 1.1, 1.2]])
        self.w2 = np.array([[0.2, 0.3],
                            [0.4, 0.5],
                            [0.6, 0.7],
                            [0.8, 0.9]])
        self.y1 = np.array([0, 1])
        self.y2 = np.array([1, 0])
        self.z1 = None
        self.a1 = None
        self.d1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, x, w1, w2):
      self.z1 = x @ w1 
      self.a1 = relu(self.z1) # activation
      
      if self.prob == 1: # (case 1) Get rid of every connection (keep_prob = 0)
        self.d1 = np.zeros_like(self.a1)
      elif self.prob == 0: # (case 2) No rid of any connection (keep_prob = 1)
        self.d1 = self.a1
      else: # (case 3) Get rid of prob amount of connection
        mask = (np.random.rand(*self.a1.shape) > self.prob) / (1.0 - self.prob)
        self.d1 = mask
      
      self.a1 = np.multiply(self.d1, self.a1)
      self.z2 = self.a1 @ w2
      self.a2 = softmax(self.z2)
      return self.a2

    def backward(self, w1, w2):
      for x, y in zip([self.x1, self.x2], [self.y1, self.y2]):
        x = x.reshape(1, -1) if x.ndim == 1 else x
        a1 = self.a1.reshape(1, -1) if self.a1.ndim == 1 else self.a1

        dL_dy_pred = (self.a2 - y).reshape(1, -1) # Gradient of Cross entropy and Softmax
        da = (self.z1 > 0).astype(float) # Gradient of ReLU in hidden layer
        dw1 = x.T @ ((dL_dy_pred @ w2.T) * da) # Gradinet of w1 using chain rule
        dw2 = a1.T @ dL_dy_pred # a1.T shape is (4, 1), dL_dy_pred shape is (1, 2)
      return dw1, dw2

# Create instance
dropout = Dropout(prob=0.4)

for x, y in zip([x1, x2], [y1, y2]):
  w1, w2 = w1_initial, w2_initial
  print(f"The loss of {x}:\n")

  for epoch in range(num_epochs):
    # Forward Pass
    y_pred = dropout.forward(x, w1, w2)
    loss = cross_entropy(y_pred, y)

    # Print loss every 10th epoch
    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{num_epochs}, Loss = {loss.item():.4f}")

    # Backward Pass
    dw1, dw2 = dropout.backward(w1, w2)
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2


  print(f"\nThe final weight of {x} is")
  print(f"w1\n{w1}")
  print(f"w2\n{w2}")
  print("\n")

```

### pytorch

```python
num_epochs = 100
learning_rate = 0.01
prob = 0.4

w1_initial = w1.clone()
w2_initial = w2.clone()

# [Function] Dropout: To preventing overfitting, make some nodes' output 0 randomly in training
class Dropout:
    def __init__(self, prob = prob): # prob: the probability to remove neuron(1: remove, 0: keep)
        assert 0 <= prob <= 1 # Dropout probability must be in range [0, 1]
        self.prob = prob
        self.x1 = torch.tensor([1.0, 2.0, 3.0])
        self.x2 = torch.tensor([4.0, 5.0, 6.0])
        self.w1 = torch.tensor([[0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.9, 1.0, 1.1, 1.2]])
        self.w2 = torch.tensor([[0.2, 0.3],
                            [0.4, 0.5],
                            [0.6, 0.7],
                            [0.8, 0.9]])
        self.y1 = torch.tensor([0, 1])
        self.y2 = torch.tensor([1, 0])
        self.z1 = None
        self.a1 = None
        self.d1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, x, w1, w2):
      self.z1 = x @ w1 
      self.a1 = relu(self.z1) # activation
      
      if self.prob == 1: # (case 1) Get rid of every connection (keep_prob = 0)
        self.d1 = torch.zeros_like(self.a1)
      elif self.prob == 0: # (case 2) No rid of any connection (keep_prob = 1)
        self.d1 = self.a1
      else: # (case 3) Get rid of prob amount of connection
        mask = (torch.rand(*self.a1.shape) > self.prob) / (1.0 - self.prob)
        self.d1 = mask
      
      self.a1 = self.a1 = torch.mul(self.d1, self.a1)
      self.z2 = self.a1 @ w2
      self.a2 = softmax(self.z2)
      return self.a2

    def backward(self, w1, w2):
      for x, y in zip([self.x1, self.x2], [self.y1, self.y2]):
        x = x.reshape(1, -1) if x.ndim == 1 else x
        a1 = self.a1.reshape(1, -1) if self.a1.ndim == 1 else self.a1
        dL_dy_pred = (self.a2 - y).reshape(1, -1) # Gradient of Cross entropy and Softmax
        da = (self.z1 > 0).float() # Gradient of ReLU in hidden layer
        dw1 = x.T @ ((dL_dy_pred @ w2.T) * da) # Gradinet of w1 using chain rule
        dw2 = a1.T @ dL_dy_pred # a1.T shape is (4, 1), dL_dy_pred shape is (1, 2)
      return dw1, dw2

# Create instance
dropout = Dropout(prob=0.4)

for x, y in zip([x1, x2], [y1, y2]):
  w1, w2 = w1_initial, w2_initial
  print(f"The loss of {x}:\n")

  for epoch in range(num_epochs):
    # Forward Pass
    y_pred = dropout.forward(x, w1, w2)
    loss = cross_entropy(y_pred, y)

    # Print loss every 10th epoch
    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{num_epochs}, Loss = {loss.item():.4f}")

    # Backward Pass
    dw1, dw2 = dropout.backward(w1, w2)
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2


  print(f"\nThe final weight of {x} is")
  print(f"w1\n{w1}")
  print(f"w2\n{w2}")
  print("\n")

```




