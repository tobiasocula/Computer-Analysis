from main import Symb, Funcs
import numpy as np

x = Symb("x")
y = Symb("y")
z = Symb("z")

implicit_func = z**2 - 1 - x**2 - 3*y
implicit_func.plot_implicit_3d(x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5))