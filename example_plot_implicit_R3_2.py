from main import Symb, Funcs
import numpy as np

x = Symb("x")
y = Symb("y")
z = Symb("z")

implicit_func = Funcs.sin(x + y + z)
implicit_func.plot_implicit_3d(x_range=(-5, 5), y_range=(-5, 5), z_range=(-5, 5))