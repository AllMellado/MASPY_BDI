from maspy import *
import math

class Map(Environment):
    def __init__(self, env_name=None):
        super().__init__(env_name)
        
        self.create(Percept())

class learnAgent(Agent):
    def stuff():
        pass
    
if __name__ == "__main__":
    max = 200
    #10-(9/math.log(x,10))
    for x in range(1,302):
        math.log
        a = (-2/math.log(max))*math.log(x)+3
        print(f"for {x}: {a}")
    
    #lrn = Learning()
    #lrn.set_params(5,2,0)
    #lrn.learn()
    
    #print(lrn.policy)