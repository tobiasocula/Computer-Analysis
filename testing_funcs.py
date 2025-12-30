from main import Symb, VFunc, BFunc, UFunc, Funcs

x = Symb("x")
y = Symb("y")

f = Funcs.sqrt(x ** 2 + y ** 2)
g = Funcs.sqrt(x ** 2 - y ** 2)
print(f)
print(f.eval({"x": 3, "y": 5}))
df = f.diff(x)
print(df)
print(df.eval({"x": 1, "y": 1}))

v = VFunc(f, g)
print(v)
norm = v.norm()
print(norm)