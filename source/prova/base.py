class A:
    def __init__(self):
        self.constA = 1

class B:
    def __init__(self):
        self.constB = 2

class C(A):
    def __init__(self,
                 Bobj: B):
        self.B = Bobj
        self.Bbis = B()

class D(B):
    def __init__(self,
                 Cobj: C):
        self.C = Cobj