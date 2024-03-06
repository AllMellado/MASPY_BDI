from maspy.agent import Agent
from maspy.environment import Environment

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

if __name__=="__main__":
    ag1 = dummy_agent("Ag1")
    ag1.connect_to(simple_env("s_env"))
    ag1.add("belief","make_action","s_env")
    ag1.start()
    
    


    
    
    