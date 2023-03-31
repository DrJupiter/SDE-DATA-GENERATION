import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class SDE():

    def __init__(self) -> None:
        pass

    def p_xt_x0(self, xt, x0, t):
        raise NotImplementedError("Conditional Distribution p(xt|x0)")
    
    def sample(self, t, x0, key):
        raise NotImplementedError("Sample x(t) given x(0)")

    def score(self, x0, t, xt):
        raise NotImplementedError("The Score of p(x(t),x(0)) i.e ∇_x(t) log(p(x(t), x(0)))") 

    def drift(self, x,t):
        raise NotImplementedError("The drift coeffecient, f(x,t) in dx = f(x,t)dt + L(x,t)db(t)") 

    def diffusion(self, x,t):
        raise NotImplementedError("The diffusion coeffecient L(x,t) in dx = f(x,t)dt + L(x,t)db(t)") 
    
    def reverse_drift(self, x, t, sm):
        raise NotImplementedError("The reverse drift term")
        
    def reverse_diffusion(self, x, t, sm):
        raise NotImplementedError("The reverse diffusion term")