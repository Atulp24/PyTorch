# We create two tensors x and y with requires_grad=True, indicating that we want to compute gradients for these tensors.
# We perform simple operations on x and y to obtain z.
# Computing Gradients: We call z.backward() to compute the gradients of z with respect to x and y. The gradients are stored in the grad attribute of each tensor.

# In the following example:

# The operation is  z=x⋅y+y2 .
# The partial derivative of  z  with respect to  x  is  ∂z∂x=y .
# The partial derivative of  z  with respect to  y  is  ∂z∂y=x+2y .
# Given  x=2.0  and  y=3.0 :

# The gradient of  z  w.r.t.  x  is  3.0 .
# The gradient of  z  w.r.t.  y  is  2.0+2⋅3.0=8.0 .
# Tensors that require gradients will have their operations tracked by PyTorch's autograd engine, enabling the computation of gradients during backpropagation.

# The automatic differentiation provided by torch.autograd simplifies this process by handling the complex chain rule calculations needed for backpropagation through the entire network.

# For  ∂z∂x :

# ∂z∂x=∂z∂p∂p∂x+∂z∂q∂q∂x=1⋅y+1⋅0=y 

# For  ∂z∂y :

# ∂z∂y=∂z∂p∂p∂y+∂z∂q∂q∂y=1⋅x+1⋅2y=x+2y 

# These equations correspond to the chain rule calculations happening behind the scenes, demonstrating how PyTorch's autograd system computes gradients through the computational graph.

# For  ∂z∂x :

# ∂z∂x=∂z∂p∂p∂x+∂z∂q∂q∂x=1⋅y+1⋅0=y 

# For  ∂z∂y :

# ∂z∂y=∂z∂p∂p∂y+∂z∂q∂q∂y=1⋅x+1⋅2y=x+2y 

# These equations correspond to the chain rule calculations happening behind the scenes, demonstrating how PyTorch's autograd system computes gradients through the computational graph.

# create tensors with requires_grad=True
# import torch
# import matplotlib.pyplot as plt

# x = torch.tensor([2.0, 5.0], requires_grad=True)
# y = torch.tensor([3.0, 7.0], requires_grad=True)

# Perform some operations
# z = x * y + y ** 2

# z.retain_grad()  # Retain the gradient for z since we will call backward multiple times

# Compute the gradients
# z_sum = z.sum().backward() # Sum the elements of z and call backward to compute gradients

# print(f"Gradient of z w.r.t x: {x.grad}")
# print(f"Gradient of z w.r.t y: {y.grad}")
# print(f"Gradient of z w.r.t z: {z.grad}")
# print(f"result of the operation: z = {z.detach()}")

# from torchviz import make_dot

# Visualize the computational graph
# dot = make_dot(z, params={'x': x, 'y': y, 'z': z})
# dot.render("grad_computation_graph", format="png")

# img = plt.imread("grad_computation_graph.png")
# plt.imshow(img)
# plt.axis('off')
# plt.show()

# Detaching Tensors from Computation Graph
# The detach() method is used to create a new tensor that shares storage with the original tensor but without tracking operations. When you call detach(), it returns a new tensor that does not require gradients. This is useful when you want to perform operations on a tensor without affecting the computation graph.

# Let's detach z from the computation graph
# print("Before detaching z from computation: ", z.requires_grad)
# z_det = z.detach()
# print("After detaching z from computation: ", z_det.requires_grad)

# Can Backpropagation be performed when requires_grad=False?
# Now the same tensors x and y are created with requires_grad=False.
# When attempting to compute the gradients using z.backward(), a RuntimeError is raised because the tensors do not require gradients, and thus do not have a grad_fn.
# In this case, since requires_grad=False was used, the computation graph is essentially empty, as no gradients will be tracked.

# x = torch.tensor(2.0, requires_grad=False)
# y = torch.tensor(3.0, requires_grad=False)


# # Perform simple operations
# z = x * y + y**2


# # Compute the gradients
# z.backward() # This will raise a RuntimeError because the tensors do not require gradients, and thus do not have a grad_fn.