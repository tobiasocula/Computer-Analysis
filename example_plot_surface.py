from main import Symb, Funcs, VFunc
import numpy as np

u = Symb("u")
v = Symb("v")
par = VFunc(
    Funcs.cos(u) * Funcs.sin(v),
    Funcs.cos(u) * Funcs.cos(v),
    Funcs.sin(v)
)
par.plot_param_surface(u_range=(0, 2*np.pi), v_range=(-np.pi/2, np.pi/2))