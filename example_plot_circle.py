from main import Symb, Funcs, VFunc
import numpy as np

t = Symb("t")

par = VFunc(
    3 * Funcs.cos(t),
    3 * Funcs.sin(t)
)
par.plot_curve_2d(t_range=(0, 2*np.pi))
