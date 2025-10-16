from maspy import *

class Controller(Agent):
    def __init__(self):
        super().__init__()
        self.add(Belief("reward", 0))
        self.add(Goal("show_reward"))
        self.step = 0
    
    @pl(gain, Goal("show_reward"), Belief("reward",Any))
    def show_reward(self, src, reward):
        self.print(f"Reward: {reward}")
        self.step += 1
        self.rm(Belief("reward",reward))
        if self.step < 5:
            self.add(Belief("reward", reward+5))
        self.add(Goal("show_reward"))
        
    @pl(gain, Goal("show_reward"))#, ~Belief("reward",Any))
    def end_reward(self, src):
        self.print(f"Not printing reward anymore")
        self.stop_cycle()

if __name__ == "__main__":
    a = Controller()
    Admin().start_system()