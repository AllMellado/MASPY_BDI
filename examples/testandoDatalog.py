from maspy import *

class sample(Agent):
    @pl(gain,Goal("test",Any), # Trigger
        [Belief("check",Any),~Belief("nien",Any)], # Context
        lambda x,y: 2 < x < y ) # Condition
    def test(self,src, var1, var2):
        print(f"test1 > {var1} e {var2}")
        self.stop_cycle()
    
    @pl(gain,Goal("test",Any),Belief("check",Any))
    def test2(self,src,x,y):
        print(f"test2 > {x} e {y}")
        self.stop_cycle()

a = sample()
a.add(Goal("test",3))
a.add(Belief("check",4))
a.start()