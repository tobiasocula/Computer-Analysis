import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class Symb:
    def __init__(self, symb):
        self.symb = symb
        self.arity = 1
        self.vars = set(symb) # convenience for plotting

    def __eq__(self, other):
        return self.symb == other.symb

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __radd__(self, other):
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
    
    def __rsub__(self, other):
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
    
    def eval_point(self, p):
        # p is scalar
        return p

class CFunc:
    def __init__(self, num):
        self.num = num
        self.vars = set() # convenient for variable collection
        self.arity = 0

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
    
    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __radd__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __rsub__(self, other):
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

class Funcs:

    @classmethod
    def sin(cls, obj):
        return resolve_ufunc(np.sin, obj, "sin")
    
    @classmethod
    def cos(cls, obj):
        return resolve_ufunc(np.cos, obj, "cos")
    
    @classmethod
    def sqrt(cls, obj):
        return resolve_ufunc(np.sqrt, obj, "sqrt")
    
    @classmethod
    def ln(cls, obj):
        return resolve_ufunc(np.log, obj, "ln")

class UFunc:
    def __init__(self, func, body, funcsymb):
        self.func = func
        self.body = body
        self.funcsymb = funcsymb
        self.vars = get_vars(self) # set of symbols
        self.arity = len(self.vars)

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __radd__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __rsub__(self, other):
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
    
    def eval_point(self, p):
        # p is a tuple of size amount of vars
        assert len(p) == self.arity, AssertionError()
        return self.eval({var : pcoord for var, pcoord in zip(list(self.vars), p)})

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
        
        if self.funcsymb == "sqrt":
            return resolve_bfunc(
                lambda x,y:x/y,
                self.body.diff(var),
                BFunc(lambda x,y:x*y, 2, Funcs.sqrt(self.body), "*"),
                "/"
            )

def resolve_ufunc(func, body, funcsymb):

    if isinstance(body, int):
        return CFunc(func(body))
    if isinstance(body, CFunc):
        return CFunc(func(body.num))
    if isinstance(body, Symb):
        return UFunc(func, body, funcsymb)
    if isinstance(body, UFunc):
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
        return {exp.symb}
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
        self.vars = get_vars(self) # set of symbols
        self.arity = len(self.vars)

    def __add__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __radd__(self, other):
        return resolve_bfunc(lambda x,y:x+y, self, other, "+")
    
    def __sub__(self, other):
        return resolve_bfunc(lambda x,y:x-y, self, other, "-")
    
    def __rsub__(self, other):
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
    
    def eval_point(self, p):
        assert len(p) == self.arity, AssertionError()
        return self.eval({var : pcoord for var, pcoord in zip(list(self.vars), p)})
        
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
            if isinstance(self.right, CFunc):
                return resolve_bfunc(
                    lambda x,y:x*y,
                    self.right,
                    resolve_bfunc(
                        lambda x,y:x*y,
                        self.left ** CFunc(self.right.num - 1),
                        self.left.diff(var),
                        "*"
                    ),
                    "*"
                )
            if isinstance(self.left, CFunc):
                return resolve_bfunc(
                    lambda x,y:x*y,
                    self,
                    Funcs.ln(self.left),
                    "*"
                )

def accumulate(null, op, lst):
    if lst == []:
        return null
    return op(lst[0], accumulate(null, op, lst[1:]))

class VFunc:
    def __init__(self, *funcs):

        # funcs are cfuncs, symbols, bfuncs or ufuncs

        self.funcs = funcs
        self.dim = len(funcs)

        # set of symbol objects
        self.vars = accumulate(set(), lambda x, y: x | y, [f.vars for f in self.funcs])
        self.arity = len(self.vars)

    def diff(self, var):
        return VFunc(*[f.diff(var) for f in self.funcs])
    
    def norm(self):
        res = CFunc(0)
        for func in self.funcs:
            res = res + func ** 2
        return Funcs.sqrt(res)

    def __str__(self):
        s = "("
        for f in self.funcs:
            s += f.__str__() + ", "
        
        return s[:-2] + ")"

    def eval(self, vars):
        return tuple([f.eval(vars) for f in self.funcs])
    
    def eval_point(self, p):
        assert all([len(p) >= k.arity for k in self.funcs]), AssertionError(f"len p: {len(p)}, arities: {[k.arity for k in self.funcs]}")
        return self.eval({var : pcoord for var, pcoord in zip(list(self.vars), p)})
    
    def __add__(self, other):
        assert self.dim == other.dim, AssertionError()
        return VFunc(*[
            self.funcs[i] + other.funcs[i]
            for i in range(self.dim)
        ])
    
    def __radd__(self, other):
        assert self.dim == other.dim, AssertionError()
        return VFunc(*[
            self.funcs[i] + other.funcs[i]
            for i in range(self.dim)
        ])
    
    def __neg__(self):
        return VFunc(*[
            self.funcs[i].__neg__()
            for i in range(self.dim)
        ])
    
    def __sub__(self, other):
        assert self.dim == other.dim, AssertionError()
        return VFunc(*[
            self.funcs[i] - other.funcs[i]
            for i in range(self.dim)
        ])
    
    def __rsub__(self, other):
        assert self.dim == other.dim, AssertionError()
        return VFunc(*[
            self.funcs[i] - other.funcs[i]
            for i in range(self.dim)
        ])
    
    def __mul__(self, other):
        assert isinstance(other, (int, CFunc)), AssertionError()
        return VFunc(*[f * other for f in self.funcs])

    def __rmul__(self, other):
        assert isinstance(other, (int, CFunc)), AssertionError()
        return VFunc(*[f * other for f in self.funcs])
    
    def cross_prod(self, other):
        # assume f1 and f2 are VFuncs with dim = 3

        # are symb, UFunc, CFunc, BFunc
        F1, F2, F3 = self.funcs
        G1, G2, G3 = other.funcs

        return VFunc(
            F2 * G3 - G2 * F3,
            G1 * F3 - F1 * G3,
            F1 * G2 - G1 * F2
        )
    
    def innerprod(self, other):
        # assume f1 and f2 have same dim
        result = CFunc(0)
        for f1, f2 in zip(self.funcs, other.funcs):
            result = result + f1 * f2

        return result

class Surface:

    def __init__(self, paramf):
        assert isinstance(paramf, VFunc), AssertionError()
        assert paramf.dim == 3 and paramf.arity <= 2, AssertionError()
        self.paramf = paramf

        # assume first and second variable are in alphabetical order
        self.vars_list = sorted(list(self.paramf.vars))

        self.df_1 = paramf.diff(Symb(self.vars_list[0]))
        self.df_2 = paramf.diff(Symb(self.vars_list[1]))

        # not normalized
        self.normal_vector = self.df_1.cross_prod(self.df_2)

        self.normal_vector_norm = self.normal_vector / self.normal_vector.norm()
        
    def tangent_plane_cartesian(self, p):
        
        # p is a point in coordinate space
        coordinate_vector = VFunc(
            Symb("x"),
            Symb("y"),
            Symb("z")
        )

        F_p = self.normal_vector.eval_point(p)
        F_p = VFunc(*[CFunc(k) for k in F_p])

        return coordinate_vector.innerprod(F_p)
    
    def tangent_plane_param(self, p):

        # p is a point in coordinate space
        # are tuples
        df1_p = self.df_1.eval_point(p)
        df2_p = self.df_2.eval_point(p)

        p_u = int(p[0])
        p_v = int(p[1])

        u_symb = Symb(self.vars_list[0])
        v_symb = Symb(self.vars_list[1])

        # tuple
        F_p = self.paramf.eval_point(p)

        return VFunc(
            CFunc(F_p[0]) + (u_symb - p_v) * CFunc(df1_p[0]) + (v_symb - p_u) * CFunc(df2_p[0]),
            CFunc(F_p[1]) + (u_symb - p_v) * CFunc(df1_p[1]) + (v_symb - p_u) * CFunc(df2_p[1]),
            CFunc(F_p[2]) + (u_symb - p_v) * CFunc(df1_p[2]) + (v_symb - p_u) * CFunc(df2_p[2]),
        )
    
    def show(self, u_range, v_range, nu=200, nv=200, p_tangent_plane=None):
        
        # Parameter grids
        u = np.linspace(u_range[0], u_range[1], nu)
        v = np.linspace(v_range[0], v_range[1], nv)

        U, V = np.meshgrid(u, v)
        
        # Extract parameter symbols
        vars = list(self.paramf.vars)
        u_symb = vars[0]
        v_symb = vars[1]
        
        # Vectorized evaluation
        X = np.vectorize(lambda ui, vi: self.paramf.funcs[0].eval({u_symb: ui, v_symb: vi}))(U, V)
        Y = np.vectorize(lambda ui, vi: self.paramf.funcs[1].eval({u_symb: ui, v_symb: vi}))(U, V)
        Z = np.vectorize(lambda ui, vi: self.paramf.funcs[2].eval({u_symb: ui, v_symb: vi}))(U, V)
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, alpha=0.8, linewidth=0, antialiased=True)

        if p_tangent_plane is not None:

            plane = self.tangent_plane_param(p_tangent_plane)

            X = np.vectorize(lambda ui, vi: plane.funcs[0].eval({u_symb: ui, v_symb: vi}))(U, V)
            Y = np.vectorize(lambda ui, vi: plane.funcs[1].eval({u_symb: ui, v_symb: vi}))(U, V)
            Z = np.vectorize(lambda ui, vi: plane.funcs[2].eval({u_symb: ui, v_symb: vi}))(U, V)

            ax.plot_surface(X, Y, Z, alpha=0.8, linewidth=0, antialiased=True)

        plt.show()

class Curve2D:

    def __init__(self, paramf):

        assert isinstance(paramf, VFunc), AssertionError()
        assert paramf.dim == 2 and paramf.arity <= 1, AssertionError()
        self.paramf = paramf

        # assume first and second variable are in alphabetical order
        self.var_string_symb = list(self.paramf.vars)[0] # only one var

        self.df_vector = paramf.diff(Symb(self.var_string_symb))
        self.curv_vector = self.df_vector.diff(Symb(self.var_string_symb))
        
        df_vect_norm = self.df_vector.norm()
        curv_vect_norm = self.curv_vector.norm()
        self.curv = Funcs.sqrt(
            df_vect_norm ** 2 * curv_vect_norm ** 2
            - self.df_vector.innerprod(self.curv_vector) ** 2
        ) / df_vect_norm ** 3

    def tangent_line_vect(self, p):

        # p is a scalar
        dir_vect = self.df_vector.eval_point([p])

        symb = Symb(self.var_string_symb)

        # scalar 
        F_p = self.paramf.eval_point([p])

        return VFunc(
            CFunc(F_p[0]) + (symb - CFunc(int(p))) * CFunc(dir_vect[0]),
            CFunc(F_p[1]) + (symb - CFunc(int(p))) * CFunc(dir_vect[1]),
        )

    def show(self, t_range, nt=200, tangent_line_p=None):

        # Parameter grids
        t = np.linspace(t_range[0], t_range[1], nt)
        
        # Vectorized evaluation
        X = np.vectorize(lambda ti: self.paramf.funcs[0].eval({self.var_string_symb: ti}))(t)
        Y = np.vectorize(lambda ti: self.paramf.funcs[1].eval({self.var_string_symb: ti}))(t)
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
        ax.plot(X, Y)

        if tangent_line_p is not None:

            line = self.tangent_line_vect(tangent_line_p)

            X = np.vectorize(lambda ti: line.funcs[0].eval({self.var_string_symb: ti}))(t)
            Y = np.vectorize(lambda ti: line.funcs[1].eval({self.var_string_symb: ti}))(t)

            ax.plot(X, Y)

        plt.show()

class Curve3D:

    def __init__(self, paramf):

        assert isinstance(paramf, VFunc), AssertionError()
        assert paramf.dim == 3 and paramf.arity <= 1, AssertionError()
        self.paramf = paramf

        # assume first and second variable are in alphabetical order
        self.var_string_symb = list(self.paramf.vars)[0] # only one var

        self.df_vector = paramf.diff(Symb(self.var_string_symb))

        double_df = self.df_vector.diff(Symb(self.var_string_symb))
        triple_df = double_df.diff(Symb(self.var_string_symb))
        t = double_df.cross_prod(self.df_vector).norm()

        # from formula found at https://en.wikipedia.org/wiki/Curvature
        self.curv = t / (self.df_vector.norm() ** 3)
        
        # determinant
        self.torsion = (
            self.df_vector[0] * (double_df.funcs[1] * triple_df.funcs[2] - double_df.funcs[2] * triple_df.funcs[1])
            - self.df_vector[1] * (double_df.funcs[0] * triple_df.funcs[2] - double_df.funcs[2] * triple_df.funcs[0])
            + self.df_vector[2] * (double_df.funcs[0] * triple_df.funcs[1] - double_df.funcs[1] * triple_df.funcs[0])
        ) / (t ** 2)

    def tangent_line_vect(self, p):

        # p is a scalar
        dir_vect = self.df_vector.eval_point([p])

        symb = Symb(self.var_string_symb)

        # scalar 
        F_p = self.paramf.eval_point([p])

        return VFunc(
            CFunc(F_p[0]) + (symb - CFunc(int(p))) * CFunc(dir_vect[0]),
            CFunc(F_p[1]) + (symb - CFunc(int(p))) * CFunc(dir_vect[1]),
            CFunc(F_p[2]) + (symb - CFunc(int(p))) * CFunc(dir_vect[2])
        )
    
    def show(self, t_range, nt=200, tangent_line_p=None):

        # Parameter grids
        t = np.linspace(t_range[0], t_range[1], nt)
        
        # Vectorized evaluation
        X = np.vectorize(lambda ti: self.paramf.funcs[0].eval({self.var_string_symb: ti}))(t)
        Y = np.vectorize(lambda ti: self.paramf.funcs[1].eval({self.var_string_symb: ti}))(t)
        Z = np.vectorize(lambda ti: self.paramf.funcs[2].eval({self.var_string_symb: ti}))(t)
        
        # Plot 3D surface
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')
        ax.plot(X, Y, Z)

        if tangent_line_p is not None:

            line = self.tangent_line_vect(tangent_line_p)

            X = np.vectorize(lambda ti: line.funcs[0].eval({self.var_string_symb: ti}))(t)
            Y = np.vectorize(lambda ti: line.funcs[1].eval({self.var_string_symb: ti}))(t)
            Z = np.vectorize(lambda ti: line.funcs[2].eval({self.var_string_symb: ti}))(t)

            ax.plot(X, Y, Z)

        plt.show()
