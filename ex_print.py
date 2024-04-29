from typing import TypeVar
from maspy.agent import *
from maspy.environment import Environment, Percept
from maspy.utils import utils as dt

class dummy_agent(Agent):
    def __init__(self, agent_name):
        super().__init__(agent_name, full_log=True)
    
    @pl(gain, Belief("make_action",("Name",)))
    def action_on_env(self, src, env_name):
        self.execute_in(env_name).env_action(self)
        self.stop_cycle()
        
class simple_env(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        
    def env_action(self, src):
        self.print(f"Action by {src.my_name}")

broadcast = TypeVar('broadcast')

if __name__=="__main__":
    ag1 = dummy_agent("Ag1")
    ag1.connect_to(simple_env("s_env"))
    ag1.add(Belief("make_action",("s_env",)))
    ag1.start()
    
    


    
    
    