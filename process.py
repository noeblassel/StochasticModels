class StochasticProcess:

    '''The base class for the StochasticProcess hierarchy.
        Every StochasticProcess has a "dim" attribute, which stores the topological dimension of the state space.
        Every StochasticProcess object implements a "sample" method, which returns a NumPy array of sample points,
        which represents an approximate sample path for this process.'''

    def __init__(self,dim):
        self.dim=dim

    def sample(self,*args):
        raise NotImplementedError("Calling sample() on base class StochasticProcess.")

class Deterministic(StochasticProcess):
    def __init__(self,F,dim=0):
        super(Deterministic,self).__init__(dim)
        self.F=F
    
    def __call__(self,*args):
        return self.F(*args)
    
    def sample(self,*args):
        return self.F(*args)