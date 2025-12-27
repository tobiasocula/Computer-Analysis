from main import Symb, Funcs, VFunc
import numpy as np

t = Symb("t")

par = VFunc(
    Funcs.cos(t),
    Funcs.sin(t),
    t
)
par.plot_curve_3d(t_range=(0, 4*np.pi))
