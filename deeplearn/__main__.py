# Use batch normalization.

import numpy
import math
import matplotlib.pyplot as pyplot

radians = numpy.linspace(-math.pi, math.pi)
theta_2d, phi_2d = numpy.meshgrid(radians, radians)
theta = theta_2d.ravel()
phi = phi_2d.ravel()
x = numpy.sin(theta) * numpy.cos(phi)
y = numpy.sin(theta) * numpy.sin(phi)
z = numpy.cos(theta)

figure = pyplot.figure()
axes = figure.add_subplot(projection="3d")
axes.scatter(x, y, z)
axes.set_aspect('equal')
pyplot.show()
