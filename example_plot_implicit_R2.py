from main import Symb, Funcs
import numpy as np

x = Symb("x")
y = Symb("y")

implicit_func = y + x**2 + 2*x - 3*x**3
implicit_func.plot_implicit_2d(x_range=(-10, 10), y_range=(-20, 5))