from maspy.agent import Agent
from maspy.environment import Environment, Percept
from maspy.utils import utils as dt

class dummy_agent(Agent):
    def __init__(self, agent_name):
        super().__init__(agent_name, full_log=True)
    
    @Agent.plan("make_action")
    def action_on_env(self, src, env_name):
        self.execute_in(env_name).env_action(self)
        self.stop_cycle()
        
class simple_env(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        
    def env_action(self, src):
        self.print(f"Action by {src.my_name}")

def test(x):
    x = 10

if __name__=="__main__":
    a = 5
    test(a)
    print(a)
    #ag = Agent("Simple",full_log=True)
    #ag2 = Agent("Simple",full_log=True)
    #env = Environment("Adding")
    #ag.connect_to(env)
    #print(env.agent_list,"\n",ag.my_name)
    #ag.send(ag2.my_name,"tell",("hue",))
    #env = Environment()
    #for i in range(10):
    #    env.create_percept("Spot",i)
    #for i in range(10):
    #    env.create_percept("Spot",i,"VIP")
    #env.print_percepts
    #print("\n",env.perception())
    
    #ag1 = dummy_agent("Ag1")
    #ag1.connect_to(simple_env("s_env"))
    #ag1.add("belief","make_action","s_env")
    #ag1.start()
    
    


    
    
    