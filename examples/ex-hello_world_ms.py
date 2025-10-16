from maspy import *
import sys
import json

class HelloAgent(Agent):
    @pl(gain,Belief("hello"))
    def func(self, src):
        pass # self.print(f"Hello World! I'm agent {self.my_name}")
        self.stop_cycle()

def main(num_agents: int = 10):
    [HelloAgent(beliefs=Belief("hello")) for _ in range(num_agents)]
    Admin().start_system()
    
if __name__ == "__main__":
    num_agents = int(sys.argv[1])
    with open(r"T:\Downloads\measuring\duration.json","w") as f:
        json.dump({"running": "hello-world", "num_agents": sys.argv[1]}, f)
    [HelloAgent(beliefs=Belief("hello")) for _ in range(num_agents)]
    Admin().start_system()
    with open(r"T:\Downloads\measuring\duration.json","w") as f:
        json.dump({"elapsed_time": Admin().elapsed_time}, f)
    