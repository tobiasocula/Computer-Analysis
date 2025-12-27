import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Symb:
    def __init__(self, symb):
        self.symb = symb
        self.arity = 1
        self.vars = set(symb) # convenience for plotting

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __rmul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __mul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __neg__(self):
        return resolve_ufunc(lambda x:-x, self, "-")
    
    def __truediv__(self, other):
        return resolve_bfunc(lambda x,y:x/y, self, other, "/")
    
    def __pow__(self, other):
        return resolve_bfunc(lambda x,y:x**y, self, other, "^")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __str__(self):
        return self.symb
    
    def diff(self, var):
        if var.symb == self.symb:
            return CFunc(1)
        return CFunc(0)
    
    def eval(self, vars):
        if self.symb in vars.keys():
            return vars[self.symb]
        raise AssertionError()

class CFunc:
    def __init__(self, num):
        self.num = num

    def __str__(self):
        return str(self.num)
    
    def diff(self, var):
        return CFunc(0)
    
    def isnull(self):
        return self.num == 0
    
    def isone(self):
        return self.num == 1
    
    def eval(self, vars):
        return self.num

class Funcs:

    @classmethod
    def sin(cls, obj):
        return resolve_ufunc(np.sin, obj, "sin")
    
    @classmethod
    def cos(cls, obj):
        return resolve_ufunc(np.cos, obj, "cos")

class UFunc:
    def __init__(self, func, body, funcsymb):
        self.func = func
        self.body = body
        self.funcsymb = funcsymb
        self.vars = get_vars(self)
        self.arity = len(self.vars)

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __rmul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __mul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __neg__(self):
        return resolve_ufunc(lambda x:-x, self, "-")
    
    def __truediv__(self, other):
        return resolve_bfunc(lambda x,y:x/y, self, other, "/")
    
    def __pow__(self, other):
        return resolve_bfunc(lambda x,y:x**y, self, other, "^")
    
    def eval(self, vars):
        return self.func(self.body.eval(vars))
    
    def __str__(self):
        if isinstance(self.body, (UFunc, BFunc)):
            return f"{self.funcsymb}({self.body.__str__()})"
        return f"{self.funcsymb}({self.body})"
    
    def diff(self, var):

        if self.funcsymb == "sin":
            return resolve_bfunc(
                lambda x,y:x*y,
                UFunc(np.cos, self.body, "cos"),
                self.body.diff(var),
                "*"
            )
        if self.funcsymb == "cos":
            return resolve_bfunc(
                lambda x,y:x*y,
                -UFunc(np.sin, self.body, "sin"),
                self.body.diff(var),
                "*"
            )
        
    def plot_implicit_3d(self, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), z_range=(-1.5, 1.5), 
           nx=50, ny=50, nz=50):
        assert self.arity == 3, AssertionError()
        
        # Create 3D grid
        x = np.linspace(*x_range, nx)
        y = np.linspace(*y_range, ny)
        z = np.linspace(*z_range, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Vectorized evaluation on 3D grid
        F = np.vectorize(lambda xi, yi, zi: self.eval({'x': xi, 'y': yi, 'z': zi}))(X, Y, Z)
        
        # Marching cubes to extract isosurface f=0
        
        try:
            verts, faces, normals, values = measure.marching_cubes(F, level=0, spacing=(1,1,1))
        except ValueError as e:
            print(f"Marching cubes failed: {e}")
            print("Try adjusting grid resolution or ranges")
            return
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh from marching cubes output
        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidths=0.5, edgecolors='gray')
        mesh.set_facecolor((0.7, 0.8, 1))  # Light blue
        ax.add_collection3d(mesh)
        
        # Set limits based on grid
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Implicit surface: {self.__str__()} = 0')
        
        plt.show()

    def plot_implicit_2d(self, x_range=(-1, 1), y_range=(-1, 1), nx=200, ny=200):

        assert self.arity == 2, AssertionError()
        x = np.linspace(*x_range, nx)
        y = np.linspace(*y_range, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda xi, yi: self.eval({'x': xi, 'y': yi}))(X, Y)
        plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Implicit curve: {self.__str__()} = 0')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()
    
def resolve_ufunc(func, body, funcsymb):

    if isinstance(body, int):
        return CFunc(func(body))
    if isinstance(body, CFunc):
        return CFunc(func(body.num))
    if isinstance(body, Symb):
        return UFunc(func, body, funcsymb)
    if isinstance(body, UFunc): # sin(body) = sin(cos(body.body))
        body_res = resolve_ufunc(body.func, body.body, body.funcsymb)
        return UFunc(func, body_res, funcsymb)
    # body is BFunc
    body_res = resolve_bfunc(body.func, body.left, body.right, body.funcsymb)
    return UFunc(func, body_res, funcsymb)
        
def resolve_bfunc(func, left, right, funcsymb):

    if funcsymb == "+":
        
        if (
            isinstance(left, int) and left == 0
            or
            isinstance(left, CFunc) and left.isnull()
        ):
            # only returning right
            if isinstance(right, int):
                return CFunc(right)
            if isinstance(right, (Symb, CFunc)):
                return right
            if isinstance(right, UFunc):
                return resolve_ufunc(right.func, right.body, right.funcsymb)
            # right is BFunc
            return resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        
        # left non zero
        if (
            isinstance(right, int) and right == 0
            or
            isinstance(right, CFunc) and right.isnull()
        ):
            # right is zero
            if isinstance(left, int):
                return CFunc(left)
            if isinstance(left, (Symb, CFunc)):
                return left
            if isinstance(left, UFunc):
                return resolve_ufunc(left.func, left.body, left.funcsymb)
            # right is BFunc
            return resolve_bfunc(left.func, left.left, left.right, left.funcsymb)
        
        # left and right non zero
        if isinstance(left, int):

            if isinstance(right, int):
                return CFunc(left + right)
            if isinstance(right, CFunc):
                return CFunc(left + right.num)
            if isinstance(right, Symb):
                return BFunc(func, CFunc(left), right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, CFunc(left), right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, CFunc(left), right_res, funcsymb)
        
        if isinstance(left, CFunc):

            if isinstance(right, int):
                return CFunc(left.num + right)
            if isinstance(right, CFunc):
                return CFunc(left.num + right.num)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, Symb):

            if isinstance(right, int):
                return BFunc(func, left, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, Symb):
                if left.symb == right.symb:
                    return BFunc(lambda x,y:x*y, CFunc(2), right, "*")
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, UFunc):

            left_res = resolve_ufunc(left.func, left.body, left.funcsymb)

            if isinstance(right, int):
                return BFunc(func, left_res, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, Symb):
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left_res, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        
        # left is BFunc

        left_res = resolve_bfunc(left.func, left.left, left.right, left.funcsymb)

        if isinstance(right, int):
            return BFunc(func, left_res, CFunc(right), funcsymb)
        if isinstance(right, CFunc):
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, Symb):
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, UFunc):
            right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        # right is BFunc
        right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        return BFunc(func, left_res, right_res, funcsymb)
    
    if funcsymb == "-":

        if (
            isinstance(left, int) and left == 0
            or
            isinstance(left, CFunc) and left.isnull()
        ):
            # only returning right
            if isinstance(right, int):
                return CFunc(-right)
            if isinstance(right, (Symb, CFunc)):
                return -right
            if isinstance(right, UFunc):
                return -resolve_ufunc(right.func, right.body, right.funcsymb)
            # right is BFunc
            return -resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        
        # left non zero
        if (
            isinstance(right, int) and right == 0
            or
            isinstance(right, CFunc) and right.isnull()
        ):
            # right is zero
            if isinstance(left, int):
                return CFunc(left)
            if isinstance(left, (Symb, CFunc)):
                return left
            if isinstance(left, UFunc):
                return resolve_ufunc(left.func, left.body, left.funcsymb)
            # right is BFunc
            return resolve_bfunc(left.func, left.left, left.right, left.funcsymb)
        
        # left and right non zero
        if isinstance(left, int):

            if isinstance(right, int):
                return CFunc(left - right)
            if isinstance(right, CFunc):
                return CFunc(left - right.num)
            if isinstance(right, Symb):
                return BFunc(func, CFunc(left), right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, CFunc(left), right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, CFunc(left), right_res, funcsymb)
        
        if isinstance(left, CFunc):

            if isinstance(right, int):
                return CFunc(left.num - right)
            if isinstance(right, CFunc):
                return CFunc(left.num - right.num)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, Symb):

            if isinstance(right, int):
                return BFunc(func, left, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, Symb):
                if left.symb == right.symb:
                    return CFunc(0)
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, UFunc):

            left_res = resolve_ufunc(left.func, left.body, left.funcsymb)

            if isinstance(right, int):
                return BFunc(func, left_res, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, Symb):
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                return BFunc(func, left_res, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        
        # left is BFunc

        left_res = resolve_bfunc(left.func, left.left, left.right, left.funcsymb)

        if isinstance(right, int):
            return BFunc(func, left_res, CFunc(right), funcsymb)
        if isinstance(right, CFunc):
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, Symb):
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, UFunc):
            right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        # right is BFunc
        right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        return BFunc(func, left_res, right_res, funcsymb)
    
    if funcsymb == "*":
        
        if isinstance(left, int):

            if left == 0:
                return CFunc(0)

            if isinstance(right, int):
                return CFunc(left * right)
            if isinstance(right, CFunc):
                return CFunc(left * right.num)
            if isinstance(right, Symb):
                if left == 1:
                    return right
                return BFunc(func, CFunc(left), right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(0)
                    return CFunc(right_res.num * left)
                if left == 1:
                    return right_res
                return BFunc(func, CFunc(left), right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(0)
                return CFunc(right_res.num * left)
            if left == 1:
                return right_res
            return BFunc(func, CFunc(left), right_res, funcsymb)
        
        if isinstance(left, CFunc):

            if left.isnull():
                return CFunc(0)

            if isinstance(right, int):
                return CFunc(left.num * right)
            if isinstance(right, CFunc):
                return CFunc(left.num * right.num)
            if isinstance(right, Symb):
                if left.isone():
                    return right
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(0)
                    if left.isone():
                        return right_res
                    return BFunc(func, left, right_res, funcsymb)
                if left.isone():
                    return right_res
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(0)
                return CFunc(right_res.num * left.num)
            if left.isone():
                return right_res
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, Symb):
            
            if isinstance(right, int):
                if right == 1:
                    return left
                if right == 0:
                    return CFunc(0)
                return BFunc(func, left, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                if right.isone():
                    return left
                if right.isnull():
                    return CFunc(0)
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(0)
                    if right_res.isone():
                        return left
                    return BFunc(func, left, right_res, funcsymb)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(0)
                if right_res.isone():
                    return left
                return BFunc(func, left, right_res, funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, UFunc):
            left_res = resolve_ufunc(left.func, left.body, left.funcsymb)
            if isinstance(left_res, CFunc) and left_res.isnull():
                return CFunc(0)
            if isinstance(right, int):
                if right == 1:
                    return left_res
                if right == 0:
                    return CFunc(0)
                if isinstance(left_res, CFunc) and left_res.isone():
                    return CFunc(right)
                return BFunc(func, left_res, CFunc(right), funcsymb)
            if isinstance(right, Symb):
                if isinstance(left_res, CFunc) and left_res.isone():
                    return right
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, CFunc):
                if right.isnull():
                    return CFunc(0)
                if right.isone():
                    return left_res
                if isinstance(left_res, CFunc) and left_res.isone():
                    return right
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(left_res, CFunc) and left_res.isone():
                    return right_res
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(0)
                    if right_res.isone():
                        return left_res
                    return BFunc(func, left_res, right_res, funcsymb)
                return BFunc(func, left_res, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(left_res, CFunc) and left_res.isone():
                return right_res
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(0)
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)                
        
        # left is BFunc
        left_res = resolve_bfunc(left.func, left.left, left.right, left.funcsymb)
        if isinstance(left_res, CFunc) and left_res.isnull():
            return CFunc(0)
        if isinstance(right, int):
            if right == 1:
                return left_res
            if right == 0:
                return CFunc(0)
            if isinstance(left_res, CFunc) and left_res.isone():
                return CFunc(right)
            return BFunc(func, left_res, CFunc(right), funcsymb)
        if isinstance(right, Symb):
            if isinstance(left_res, CFunc) and left_res.isone():
                return right
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, CFunc):
            if right.isnull():
                return CFunc(0)
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, UFunc):
            right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
            if isinstance(left_res, CFunc) and left_res.isone():
                return right_res
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(0)
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)    
            return BFunc(func, left_res, right_res, funcsymb)    
        # right is BFunc
        right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        if isinstance(left_res, CFunc) and left_res.isone():
            return right_res
        if isinstance(right_res, CFunc):
            if right_res.isnull():
                return CFunc(0)
            if right_res.isone():
                return left_res
            return BFunc(func, left_res, right_res, funcsymb)
        return BFunc(func, left_res, right_res, funcsymb)
    
    if funcsymb == "/":

        # left in (int, CF, Symb, UF, BF)
        # right in (int, CF, Symb, UF, BF)

        if isinstance(left, int):

            if left == 0:
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    raise AssertionError()
                return CFunc(left / right)
            if isinstance(right, CFunc):
                if right.isnull():
                    raise AssertionError()
                return CFunc(left / right.num)
            if isinstance(right, Symb):
                return BFunc(func, CFunc(left), right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        raise AssertionError()
                    if right_res.isone():
                        return CFunc(left)
                    return CFunc(left / right_res.num)
                return BFunc(func, CFunc(left), right_res, funcsymb)
            # right is BFunc
            right_res = resolve_ufunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    raise AssertionError()
                if right_res.isone():
                    return CFunc(left)
                return CFunc(left / right_res.num)
            return BFunc(func, CFunc(left), right_res, funcsymb)
        
        if isinstance(left, CFunc):

            if left.isnull():
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    raise AssertionError()
                return CFunc(left.num / right)
            if isinstance(right, CFunc):
                if right.isnull():
                    raise AssertionError()
                return CFunc(left.num / right.num)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        raise AssertionError()
                    if right_res.isone():
                        return left
                    return CFunc(left.num / right_res.num)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_ufunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    raise AssertionError()
                if right_res.isone():
                    return left
                return CFunc(left.num / right_res.num)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, Symb):

            if isinstance(right, int):
                if right == 0:
                    raise AssertionError()
                if right == 1:
                    return left
                return BFunc(func, left, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                if right.isnull():
                    raise AssertionError()
                if right.isone():
                    return left
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, Symb):
                if right.symb == left.symb:
                    return CFunc(1)
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        raise AssertionError()
                    if right_res.isone():
                        return left
                    return BFunc(func, left, right_res, funcsymb)
                if isinstance(right_res, Symb):
                    if right_res.symb == left.symb:
                        return CFunc(1)
                    return BFunc(func, left, right_res, funcsymb)
                # right_res is ufunc or bfunc
                return BFunc(func, left, right_res, funcsymb)
            # right is Bfunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    raise AssertionError()
                if right_res.isone():
                    return left
                return BFunc(func, left, right_res, funcsymb)
            if isinstance(right_res, Symb):
                if right_res.symb == left.symb:
                    return CFunc(1)
                return BFunc(func, left, right_res, funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, UFunc):
            left_res = resolve_ufunc(left.func, left.body, left.funcsymb)
            if isinstance(left_res, CFunc) and left_res.isnull():
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    raise AssertionError()
                if right == 1:
                    return left_res
                return BFunc(func, left_res, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                if right.isnull():
                    raise AssertionError()
                if right.isone():
                    return left_res
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, Symb):
                if isinstance(left_res, Symb) and left_res.symb == right.symb:
                    return CFunc(1)
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        raise AssertionError()
                    if right_res.isone():
                        return left_res
                    return BFunc(func, left_res, right_res, funcsymb)
                if isinstance(right_res, Symb):
                    if isinstance(left_res, Symb) and left_res.symb == right_res.symb:
                        return CFunc(1)
                    return BFunc(func, left_res, right_res, funcsymb)
                return BFunc(func, left_res, right_res, funcsymb)
            # right is Bfunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    raise AssertionError()
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)
            if isinstance(right_res, Symb):
                if isinstance(left_res, Symb) and left_res.symb == right_res.symb:
                    return CFunc(1)
                return BFunc(func, left_res, right_res, funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        
        # left is Bfunc
        left_res = resolve_bfunc(left.func, left.left, left.right, left.funcsymb)
        if isinstance(left_res, CFunc) and left_res.isnull():
            return CFunc(0)
        if isinstance(right, int):
            if right == 0:
                raise AssertionError()
            if right == 1:
                return left_res
            return BFunc(func, left_res, CFunc(right), funcsymb)
        if isinstance(right, CFunc):
            if right.isnull():
                raise AssertionError()
            if right.isone():
                return left_res
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, Symb):
            if isinstance(left_res, Symb) and left_res.symb == right.symb:
                return CFunc(1)
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, UFunc):
            right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    raise AssertionError()
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)
            if isinstance(right_res, Symb):
                if isinstance(left_res, Symb) and left_res.symb == right_res.symb:
                    return CFunc(1)
                return BFunc(func, left_res, right_res, funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        # right is Bfunc
        right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        if isinstance(right_res, CFunc):
            if right_res.isnull():
                raise AssertionError()
            if right_res.isone():
                return left_res
            return BFunc(func, left_res, right_res, funcsymb)
        if isinstance(right_res, Symb):
            if isinstance(left_res, Symb) and left_res.symb == right_res.symb:
                return CFunc(1)
            return BFunc(func, left_res, right_res, funcsymb)
        return BFunc(func, left_res, right_res, funcsymb)
    

    if funcsymb == "^":

        if isinstance(left, int):

            if left == 0:
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    return CFunc(1)
                return CFunc(left ** right)
            if isinstance(right, CFunc):
                if right.isnull():
                    return CFunc(1)
                return CFunc(left ** right.num)
            if isinstance(right, Symb):
                return BFunc(func, CFunc(left), right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(1)
                    if right_res.isone():
                        return CFunc(left)
                    return CFunc(left ** right_res.num)
                return BFunc(func, CFunc(left), right_res, funcsymb)
            # right is BFunc
            right_res = resolve_ufunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(1)
                if right_res.isone():
                    return CFunc(left)
                return CFunc(left ** right_res.num)
            return BFunc(func, CFunc(left), right_res, funcsymb)
        
        if isinstance(left, CFunc):

            if left.isnull():
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    return CFunc(1)
                return CFunc(left.num ** right)
            if isinstance(right, CFunc):
                if right.isnull():
                    return CFunc(1)
                return CFunc(left.num ** right.num)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(1)
                    if right_res.isone():
                        return left
                    return CFunc(left.num ** right_res.num)
                return BFunc(func, left, right_res, funcsymb)
            # right is BFunc
            right_res = resolve_ufunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(1)
                if right_res.isone():
                    return left
                return CFunc(left.num ** right_res.num)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, Symb):

            if isinstance(right, int):
                if right == 0:
                    return CFunc(1)
                if right == 1:
                    return left
                return BFunc(func, left, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                if right.isnull():
                    return CFunc(1)
                if right.isone():
                    return left
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, Symb):
                return BFunc(func, left, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(1)
                    if right_res.isone():
                        return left
                    return BFunc(func, left, right_res, funcsymb)
                # right_res is ufunc or bfunc or symb
                return BFunc(func, left, right_res, funcsymb)
            # right is Bfunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(1)
                if right_res.isone():
                    return left
                return BFunc(func, left, right_res, funcsymb)
            return BFunc(func, left, right_res, funcsymb)
        
        if isinstance(left, UFunc):
            left_res = resolve_ufunc(left.func, left.body, left.funcsymb)
            if isinstance(left_res, CFunc) and left_res.isnull():
                return CFunc(0)
            if isinstance(right, int):
                if right == 0:
                    return CFunc(1)
                if right == 1:
                    return left_res
                return BFunc(func, left_res, CFunc(right), funcsymb)
            if isinstance(right, CFunc):
                if right.isnull():
                    return CFunc(1)
                if right.isone():
                    return left_res
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, Symb):
                return BFunc(func, left_res, right, funcsymb)
            if isinstance(right, UFunc):
                right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
                if isinstance(right_res, CFunc):
                    if right_res.isnull():
                        return CFunc(1)
                    if right_res.isone():
                        return left_res
                    return BFunc(func, left_res, right_res, funcsymb)
                return BFunc(func, left_res, right_res, funcsymb)
            # right is Bfunc
            right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(1)
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        
        # left is Bfunc
        left_res = resolve_bfunc(left.func, left.left, left.right, left.funcsymb)
        if isinstance(left_res, CFunc) and left_res.isnull():
            return CFunc(0)
        if isinstance(right, int):
            if right == 0:
                return CFunc(1)
            if right == 1:
                return left_res
            return BFunc(func, left_res, CFunc(right), funcsymb)
        if isinstance(right, CFunc):
            if right.isnull():
                return CFunc(1)
            if right.isone():
                return left_res
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, Symb):
            return BFunc(func, left_res, right, funcsymb)
        if isinstance(right, UFunc):
            right_res = resolve_ufunc(right.func, right.body, right.funcsymb)
            if isinstance(right_res, CFunc):
                if right_res.isnull():
                    return CFunc(1)
                if right_res.isone():
                    return left_res
                return BFunc(func, left_res, right_res, funcsymb)
            return BFunc(func, left_res, right_res, funcsymb)
        
        # right is Bfunc
        right_res = resolve_bfunc(right.func, right.left, right.right, right.funcsymb)
        if isinstance(right_res, CFunc):
            if right_res.isnull():
                return CFunc(1)
            if right_res.isone():
                return left_res
            return BFunc(func, left_res, right_res, funcsymb)
        return BFunc(func, left_res, right_res, funcsymb)

def get_vars(exp):
    if isinstance(exp, Symb):
        return set(exp.symb)
    if isinstance(exp, UFunc):
        return get_vars(exp.body)
    if isinstance(exp, BFunc):
        return get_vars(exp.left) | get_vars(exp.right)
    return set()

class BFunc:
    def __init__(self, func, left, right, funcsymb):
        self.func = func
        self.left = left
        self.right = right
        self.funcsymb = funcsymb
        self.vars = get_vars(self)
        self.arity = len(self.vars)

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __rmul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __mul__(self, other):
        return resolve_bfunc(lambda x,y:x*y, self, other, "*")
    
    def __neg__(self):
        return resolve_ufunc(lambda x:-x, self, "-")
    
    def __truediv__(self, other):
        return resolve_bfunc(lambda x,y:x/y, self, other, "/")
    
    def __pow__(self, other):
        return resolve_bfunc(lambda x,y:x**y, self, other, "^")

    def eval(self, vars):
        return self.func(self.left.eval(vars), self.right.eval(vars))
        
    def __str__(self):
        if isinstance(self.left, (CFunc, Symb, UFunc)):
            if isinstance(self.right, (CFunc, Symb, UFunc)):
                return f"{self.left.__str__()}{self.funcsymb}{self.right.__str__()}"
            return f"{self.left.__str__()}{self.funcsymb}({self.right.__str__()})"
        return f"({self.left.__str__()}){self.funcsymb}({self.right.__str__()})"
    
    def diff(self, var):

        if self.funcsymb == "+":
            return resolve_bfunc(self.func, self.left.diff(var), self.right.diff(var), self.funcsymb)
        if self.funcsymb == "-":
            return resolve_bfunc(self.func, self.left.diff(var), self.right.diff(var), self.funcsymb)
        if self.funcsymb == "*":
            return resolve_bfunc(
                lambda x,y:x+y,
                self.left * self.right.diff(var),
                self.right * self.left.diff(var),
                "+"
            )
        if self.funcsymb == "/":
            return resolve_bfunc(
                lambda x,y:x/y,
                self.left.diff(var) * self.right - self.left * self.right.diff(var),
                self.right ** 2,
                "/"
            )
        if self.funcsymb == "^":
            pass

    def plot_implicit_2d(self, x_range=(-1, 1), y_range=(-1, 1), nx=200, ny=200):

        assert self.arity == 2, AssertionError()
        x = np.linspace(*x_range, nx)
        y = np.linspace(*y_range, ny)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda xi, yi: self.eval({'x': xi, 'y': yi}))(X, Y)
        plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Implicit curve: {self.__str__()} = 0')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.show()

    def plot_implicit_3d(self, x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), z_range=(-1.5, 1.5), 
           nx=50, ny=50, nz=50):
        assert self.arity == 3, AssertionError()
        
        # Create 3D grid
        x = np.linspace(*x_range, nx)
        y = np.linspace(*y_range, ny)
        z = np.linspace(*z_range, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Vectorized evaluation on 3D grid
        F = np.vectorize(lambda xi, yi, zi: self.eval({'x': xi, 'y': yi, 'z': zi}))(X, Y, Z)
        
        # Marching cubes to extract isosurface f=0
        
        try:
            verts, faces, normals, values = measure.marching_cubes(F, level=0, spacing=(1,1,1))
        except ValueError as e:
            print(f"Marching cubes failed: {e}")
            print("Try adjusting grid resolution or ranges")
            return
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create mesh from marching cubes output
        mesh = Poly3DCollection(verts[faces], alpha=0.7, linewidths=0.5, edgecolors='gray')
        mesh.set_facecolor((0.7, 0.8, 1))  # Light blue
        ax.add_collection3d(mesh)
        
        # Set limits based on grid
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_zlim(0, nz)
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title(f'Implicit surface: {self.__str__()} = 0')
        
        plt.show()


def all_eq(self, lst):
    if len(lst) == 1:
        return True
    f = lst[0]
    for k in lst[1:]:
        if k != f:
            return False
    return True

def accumulate(null, op, lst):
    if lst == []:
        return null
    return op(lst[0], accumulate(null, op, lst[1:]))

class VFunc:
    def __init__(self, *funcs):
        self.funcs = funcs
        self.dim = len(funcs)

    def diff(self, var):
        return VFunc(*[f.diff(var) for f in self.funcs])

    def __str__(self):
        s = "("
        for f in self.funcs:
            s += f.__str__() + ", "
        
        return s[:-2] + ")"

    def plot_curve_2d(self, t_range=(-np.pi, np.pi), nt=1000):
        assert self.dim == 2 and all([f.arity <= 1 for f in self.funcs]), AssertionError()
        assert 1 == len(accumulate(set(), lambda x,y:x | y, [f.vars for f in self.funcs])), AssertionError()
        
        # Parameter grid
        t = np.linspace(*t_range, nt)

        t_symb = list(self.funcs[0].vars)[0]
        
        # Evaluate parametric functions
        x_vals = np.vectorize(lambda ti: self.funcs[0].eval({t_symb: ti}))(t)
        y_vals = np.vectorize(lambda ti: self.funcs[1].eval({t_symb: ti}))(t)
        
        # Plot parametric curve
        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'({self.funcs[0]}, {self.funcs[1]})')
        plt.plot([x_vals[0]], [y_vals[0]], 'go', markersize=8, label='t=start')  # Start point
        plt.plot([x_vals[-1]], [y_vals[-1]], 'ro', markersize=8, label='t=end')  # End point
        
        plt.title(f'Parametric curve: {self.__str__()}')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        plt.show()

    def plot_curve_3d(self, t_range=(-np.pi, np.pi), nt=1000):
        assert self.dim == 3 and all([f.arity <= 1 for f in self.funcs]), AssertionError()
        assert 1 == len(accumulate(set(), lambda x,y: x | y, [f.vars for f in self.funcs])), AssertionError()
        
        # Parameter grid
        t = np.linspace(*t_range, nt)
        
        t_symb = list(self.funcs[0].vars)[0]
        
        # Evaluate parametric functions
        x_vals = np.vectorize(lambda ti: self.funcs[0].eval({t_symb: ti}))(t)
        y_vals = np.vectorize(lambda ti: self.funcs[1].eval({t_symb: ti}))(t)
        z_vals = np.vectorize(lambda ti: self.funcs[2].eval({t_symb: ti}))(t)
        
        # Plot parametric curve in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(x_vals, y_vals, z_vals, 'b-', linewidth=3, label=f'({self.funcs[0]}, {self.funcs[1]}, {self.funcs[2]})')
        ax.plot([x_vals[0]], [y_vals[0]], [z_vals[0]], 'go', markersize=10, label='t=start')
        ax.plot([x_vals[-1]], [y_vals[-1]], [z_vals[-1]], 'ro', markersize=10, label='t=end')
        
        ax.set_title(f'Parametric curve: {self.__str__()}')
        
        ax.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    def plot_param_surface(self, u_range=(-np.pi, np.pi), v_range=(-np.pi, np.pi), 
                      nu=50, nv=50):
        assert self.dim == 3 and all([f.arity <= 2 for f in self.funcs]), AssertionError()
        assert 2 == len(accumulate(set(), lambda x,y: x | y, [f.vars for f in self.funcs])), AssertionError()
        
        # Parameter grids
        u = np.linspace(*u_range, nu)
        v = np.linspace(*v_range, nv)
        U, V = np.meshgrid(u, v)
        
        # Extract parameter symbols
        u_symb = list(self.funcs[0].vars)[0]
        v_symb = list(self.funcs[0].vars)[1]
        
        # Vectorized evaluation
        X = np.vectorize(lambda ui, vi: self.funcs[0].eval({u_symb: ui, v_symb: vi}))(U, V)
        Y = np.vectorize(lambda ui, vi: self.funcs[1].eval({u_symb: ui, v_symb: vi}))(U, V)
        Z = np.vectorize(lambda ui, vi: self.funcs[2].eval({u_symb: ui, v_symb: vi}))(U, V)
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0, antialiased=True)

        ax.set_title(f'Parametric surface: {self}')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
