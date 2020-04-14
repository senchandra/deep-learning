# Import libraries
import matplotlib.pyplot as plt
import numpy as np
import math

# Prepare the x axis for generating points
x = np.linspace(-5, 5, 101)

# Get slope for the line for every degree from 0 to 179
slope = [math.tan(math.pi*i/180) for i in range(180)]

# Plot the images and save it in jpg file
for i in range(len(slope)):
    plt.figure(figsize=(1,1))
    plt.axis('off')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    plt.plot(x, slope[i]*x)
    plt.savefig('images/{}.jpg'.format(i))
    plt.close()

