from main import Symb, Funcs
import numpy as np

x = Symb("x")
y = Symb("y")

implicit_func = Funcs.cos(x + 2*y)
implicit_func.plot_implicit_2d(x_range=(-10, 10), y_range=(-10, 10))