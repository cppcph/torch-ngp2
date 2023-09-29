import torch
import numpy as np
import matplotlib.pyplot as plt

# Generate some random 3D points using PyTorch
points = torch.randn(100, 3)  # 100 points in 3D

# Convert the PyTorch tensor to a NumPy array
points_np = points.numpy()

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2])

# Set labels for axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Show the plot
plt.show()