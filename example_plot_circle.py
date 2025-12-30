from main import Symb, Funcs, VFunc, Curve2D
import numpy as np

t = Symb("t")

# parametrisation
par = VFunc(
    3 * Funcs.cos(t),
    3 * Funcs.sin(t)
)

# determine point for tangent line
point = np.pi / 4

curve = Curve2D(par)
curve.show(t_range=(0, 2 * np.pi), tangent_line_p=point)
