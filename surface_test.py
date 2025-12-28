from main import Surface, VFunc, Funcs, Symb
import numpy as np

u = Symb("u")
v = Symb("v")

paramf = VFunc(
    Funcs.cos(u) * Funcs.cos(v),
    Funcs.cos(u) * Funcs.sin(v),
    Funcs.sin(u)
)
s = Surface(paramf)

p = (np.pi / 4, np.pi / 3)

print(s.normal_vector)
plane = s.tangent_plane(p)
print(plane)