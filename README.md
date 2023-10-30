# cuda-raytrace

A raytracer in cuda that can run in a colab notebook

## Run the POC in colab

```
!rm -rf cuda-raytrace
!git clone --depth 1 https://github.com/Kristoffer-Eriksson/cuda-raytrace.git
```

```
!nvcc cuda-raytrace/src/poc.cu -o poc
```

```
!./poc > random.csv
```

```py
import numpy as np
import matplotlib.pyplot as plt

# Load your CSV file
data = np.genfromtxt('random.csv', delimiter=',')

# Reshape the data into a 100x100 matrix
matrix = data.reshape((100, 100))

# Plot the matrix as an image
plt.imshow(matrix, cmap='viridis')  # You can choose other colormaps too
plt.colorbar()  # Add a colorbar for reference
plt.show()
```
