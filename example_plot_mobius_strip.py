from main import Surface, VFunc, Funcs, Symb
import numpy as np

u = Symb("u")
v = Symb("v")

R = 3

# parametrisation
paramf = VFunc(
    (R + v * Funcs.cos(u / 2)) * Funcs.cos(u),
    (R + v * Funcs.cos(u / 2)) * Funcs.sin(u),
    v * Funcs.sin(u / 2)
)
surface = Surface(paramf)

p = (np.pi / 4, np.pi / 3)

surface.show(
    u_range=(0, 2 * np.pi),
    v_range=(0, 4),
    p_tangent_plane=p
)