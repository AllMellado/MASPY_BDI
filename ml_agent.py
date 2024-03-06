from maspy.agent import Agent
from maspy.machine_learning import learning
import gymnasium as gym

class mvmt_agent(Agent):
    def __init__(self, agent_name):
        super().__init__(agent_name)
        
        # Configuring
        self.print("Configuring External learning module")
        self.ext_env = gym.make("CartPole-v1")
        obv, info = self.ext_env.reset(seed=42)
        print(self.ext_env.observation_space)
        print(f"{obv} - {info} / actions: {self.ext_env.action_space}")
        
        # Configuring
        self.print("Configuring Internal learning module")
        self.ml = learning(10,5,0.1)
        
        self.my_map = self.ml.random_map(
                        self.ml.map_size, 
                        self.ml.num_targets
                    )
        self.env = self.ml.model(self.my_map)
        self.add("B","learn_mvmt")
        
    @Agent.plan("learn_mvmt")
    def learn_to_move(self,src):
        # Learning
        self.print("Running learning")
        self.ml.learn(1) 
        self.add("B","exec_mvmt")
        
    @Agent.plan("exec_mvmt")
    def make_move(self,src):
        # Doing
        self.print("Executing")
        new_state = self.env.encode(1,1)
        for i in range(5):
            act = self.ext_env.action_space.sample()
            obv, reward, terminated, truncated, info = self.ext_env.step(act)
            self.print(f"External: Make Action <{act}> | Move to State: {obv} | Reward {reward} / {terminated} / {truncated} - {info}")
        
            action, new_state = self.ml.exec(self.env, new_state)
            self.print(f"Internal: Make Action: {action} - Move to State: {new_state}")
            new_state = new_state[-1]
        
        self.ext_env.close()
        self.stop_cycle()
        
if __name__=="__main__":
    ag = mvmt_agent("mv_ag")
    ag.start()
    

    
    