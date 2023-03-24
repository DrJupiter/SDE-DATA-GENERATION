class SDE():

    def __init__(self) -> None:
        pass

    def p_xt_x0(self, xt, x0):
        raise NotImplementedError("Conditional Distribution p(xt|x0)")
    
    def sample(self, t, x0):
        raise NotImplementedError("Sample x(t) given x(0)")

    def score(self, xt, x0):
        raise NotImplementedError("The Score of p(x(t),x(0)) i.e ∇_x(t) log(p(x(t), x(0)))") 
    