import numpy as np
import fitting
import matplotlib.pyplot as plt


# Create an array of pixel sizes (modify the range and number as needed)
pixel_sizes = np.linspace(5, 120, 100)

# Compute the localization precision for each pixel size
precision_values = fitting.localization_precision(1.1, 200, 2, pixel_sizes)
precision_values2 = fitting.localization_precision(1.1, 100, 3, pixel_sizes)


# Plot the localization precision as a function of pixel size
plt.figure(figsize=(8, 6))
plt.plot(pixel_sizes, precision_values, label='lpx, 200 photons, bg=2, and sx=126nm')
#plt.plot(pixel_sizes, precision_values2, label='lpx, 100 photons, bg=3, and sx=126nm')
plt.xlabel('Pixel Size')
plt.ylabel('Localization Precision')
plt.title('Localization Precision vs Pixel Size')
plt.legend()
plt.grid(True)
plt.show()