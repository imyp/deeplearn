"""Plot dataset."""
import deeplearn.data as data
import deeplearn.plot as plot

spheres = [pair[0] for pair in data.SphereDataset(10)]
initial_sphere = spheres.pop()
values = sum(spheres, start=initial_sphere)

plot.plot_volume(values)
