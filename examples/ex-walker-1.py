from maspy import Agent, Admin, Goal, gain, pl
from random import choice

DIRECTIONS = ["north", "south", "east", "west"]

class Walker(Agent):
    def __init__(self):
        super().__init__()
        self.add(Goal("move"))
    
    @pl(gain, Goal("move"))
    def decide_move(self,src):
        direction = choice(DIRECTIONS)
        self.print(f"Moving {direction}...")
        self.stop_cycle()
        
if __name__ == "__main__":
    #walker = Walker()
    for _ in range(10):
        arg = Walker()
    Admin().start_system()