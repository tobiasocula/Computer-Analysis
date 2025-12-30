from main import Symb, Funcs, VFunc, Surface
import numpy as np

u = Symb("u")
v = Symb("v")

# parametrisation
par = VFunc(
    Funcs.cos(u) * Funcs.cos(v),
    Funcs.cos(u) * Funcs.sin(v),
    Funcs.sin(u)
)

p = (0, 0)

surface = Surface(par)
surface.show(u_range=(-np.pi, np.pi), v_range=(-np.pi, np.pi), nu=500, nv=500,
             p_tangent_plane=p)