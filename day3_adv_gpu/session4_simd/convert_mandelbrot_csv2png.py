#!/usr/bin/env python3

import pandas
from matplotlib import pyplot as plt

mandelbrot = pandas.read_csv('./mandelbrot_set.csv',header=None)
mandelbrot[63][63] = 255
plt.imsave('mandelbrot_set.png',mandelbrot)

mandelbrot_vectorized = pandas.read_csv('./mandelbrot_set_vectorized.csv',header=None)
mandelbrot_vectorized[63][63] = 255
plt.imsave('mandelbrot_set_vectorized.png',mandelbrot_vectorized)

#from PIL import Image
#img = Image.fromarray(mandelbrot, 'I')
#img.save('my.png')
#img.show()
