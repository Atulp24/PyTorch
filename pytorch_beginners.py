import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

# print("torch version : {}".format(torch.__version__))
# print("matplotlib version : {}".format(plt.matplotlib.__version__))
# print("numpy version : {}".format(np.__version__))
# print("OpenCV version : {}".format(cv2.__version__))

# Stores pixels of the image in a 2D array
digit_0_array_og = cv2.imread("mnist_0.jpg")
digit_1_array_og = cv2.imread("mnist_1.jpg")

# digit_0_array_gray = cv2.imread("mnist_0.jpg",cv2.IMREAD_GRAYSCALE )
# digit_1_array_gray = cv2.imread("mnist_1.jpg",cv2.IMREAD_GRAYSCALE )

# # Visualize the image
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(digit_0_array_gray, cmap='gray')
# axs[0].set_title('Digit 0')
# axs[1].imshow(digit_1_array_gray, cmap='gray')
# axs[1].set_title('Digit 1')

# plt.show()

# Numpy array with three channels
# print("Image array shape:", digit_0_array_og.shape)
# print(f"Min pixel value: {np.min(digit_0_array_og)}, Max pixel value: {np.max(digit_0_array_og)}")

# Convert Numpy array to Torch tensors
# Convert the image arrays to float32 and normalize pixel values to [0, 1]
img_tensor_0 = torch.tensor(digit_0_array_og, dtype=torch.float32) / 255.0
img_tensor_1 = torch.tensor(digit_1_array_og, dtype=torch.float32) / 255.0

# print("Shape of normalized digit 0 tensor:", img_tensor_0.shape)
# print(f"Normalised Min pixel value: {torch.min(img_tensor_0)}, Max pixel value: {torch.max(img_tensor_0)}")

# plt.imshow(img_tensor_0, cmap='gray')
# plt.title('Digit 0')
# plt.axis('off')  # Turn off axis labels and ticks
# plt.show()

# Creating Input Batch
# batch_tensor = torch.stack([img_tensor_0, img_tensor_1])

# # # In PyTorch the forward pass of input images to the model is expected to have a batch_size > 1
# # print("Batch Tensor Shape:", batch_tensor.shape)  # Should print (2, H, W, C) where H is height, W is width, and C is channels

# batch_input = batch_tensor.permute(0,3,1,2)
# print("Batch Tensor Shape:", batch_input.shape)

# Create a Tensor with just ones in a column
# a = torch.ones(5)
# # Print the tensor we created
# print(a)

# Create a Tensor with just zeros in a column
# b = torch.zeros(5)
# print(b)

# We can similarly create Tensor with custom values as shown below.
# c = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
# print(c)

# Some tensors of higher dimension.
# d = torch.zeros(3,2)
# print(d)

# e = torch.ones(3,2)
# print(e)

# f = torch.tensor([[1.0, 2.0],[3.0, 4.0]])
# print(f)

# 3D Tensor
# g = torch.tensor([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
# print(g)

# We can also find out the shape of a Tensor using .shape method.
# print(d.shape)
# print(e.shape)
# print(f.shape)
# print(g.shape)

# Access an element in Tensor
# print(c[2])

# To access one particular element in a tensor, we will need to specify indices equal to the dimension of the tensor. That’s why for tensor c we only had to specify one index.

# All indices starting from 0

# Get element at row 1, column 0
# print(f[1,0])

# We can also use the following
# print(f[1][0])

# Similarly for 3D Tensor
# print(g[1,0,0])
# print(g[1][0][0])

# All elements
# print(f[:])

# All elements from index 1 to 2 (excluding element 3)
# print(c[1:3])

# All elements till index 4 (exclusive)
# print(c[:4])

# First row
# print(f[0, :])

# Second column
# print(f[:,1])

# Specify data type of elements
# int_tensor = torch.tensor([[1,2,3],[4,5,6]])
# print(int_tensor.dtype)

# What if we changed any one element to floating point number?
# int_tensor = torch.tensor([[1,2,3],[4.,5,6]])
# print(int_tensor.dtype)
# print(int_tensor)

# This can be overridden as follows
# float_tensor = torch.tensor([[1, 2, 3],[4., 5, 6]])
# int_tensor = float_tensor.type(torch.int64)
# print(int_tensor.dtype)
# print(int_tensor)

# Tensor to Array
# f_numpy = f.numpy()
# print(f_numpy)

# Array to Tensor
# h = np.array([[8,7,6,5],[4,3,2,1]])
# h_tensor = torch.from_numpy(h)
# print(h_tensor)

# Arithmetic Operations on Tensors
# Create tensor
# tensor1 = torch.tensor([[1, 2, 3], [3, 4, 5]])
# tensor2 = torch.tensor([[-1, 2, -3], [4, -5, 6]])

# Addition
# print(tensor1+tensor2)  # Element-wise addition

# We can also use
# print(torch.add(tensor1,tensor2))

# Subtraction
# print(tensor1-tensor2)
# We can also use
# print(torch.sub(tensor1,tensor2))

# Multiplication
# Tensor with Scalar
# print(tensor1 * 2)

# Tensor with another tensor
# Elementwise Multiplication
# print(tensor1 * tensor2)

# Matrix multiplication
# tensor3 = torch.tensor([[1,2],[3,4],[5,6]])
# print(torch.mm(tensor1,tensor3))

# Division
# Tensor with scalar
# print(tensor1/2)

# Tensor with another tensor
# Elementwise division
# print(tensor1/tensor2)

# Broadcasting
# a is a 1-dimensional tensor with shape ([ 3 ]).
# b is a scalar tensor with shape ([ 1 ]).
# When adding a and b, PyTorch broadcasts b to match the shape of a, resulting in ([ 1 + 4, 2 + 4, 3 + 4 ]).

# Create two 1-dimensional tensors
# a = torch.tensor([1, 2, 3])
# b = torch.tensor([4])

# adding a scalar to a vector
# result = a + b

# print("Result of Broadcasting:\n",result)

# Broadcasting allows PyTorch to perform element-wise operations on tensors of
# a is a 2-dimensional tensor with shape ([1, 3]).
# b is a 2-dimensional tensor with shape ([3, 1]).
# When adding a and b, PyTorch broadcasts both tensors to the common shape ([3, 3]), resulting in:
# ⎡⎣⎢1+4 1+5 1+6 2+4 2+5 2+6 3+4 3+5 3+6⎤⎦⎥

# Create two tensors with shapes (1, 3) and (3, 1)
# a = torch.tensor([[1, 2, 3]])
# b = torch.tensor([[4], [5], [6]])

# adding tensors of different shapes
# result = a + b
# print("Shape: ", result.shape)
# print("\n")
# print("Result of Broadcasting:\n", result)

# CPU v/s GPU Tensor
# Create a tensor for CPU
# This will occupy CPU RAM
# tensor_cpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cpu')

# Create a tensor for GPU
# This will occupy GPU RAM
# tensor_gpu = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device='cuda')

# Just like tensor creation, the operations performed for CPU and GPU tensors are also different and consume RAM corresponding to the device specified.
# This uses CPU RAM
# tensor_cpu = tensor_cpu * 5

# This uses GPU RAM
# Focus on GPU RAM Consumption
# tensor_gpu = tensor_gpu * 5

# We can move the GPU tensor to CPU and vice versa as shown below.
# Move GPU tensor to CPU
# tensor_gpu_cpu = tensor_gpu.to(device='cpu')

# Move CPU tensor to GPU
# tensor_cpu_gpu = tensor_cpu.to(device='cuda')