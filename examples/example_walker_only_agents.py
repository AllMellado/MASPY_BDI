from maspy import *
from random import choice

NUM_WALKERS = 2
DIRECTIONS = [
    "up", "down",
    "left", "right"
]

class Walker(Agent):
    def __init__(self):
        super().__init__()
        self.add(Goal("move"))

    @pl(gain, Goal("move"))
    def make_move(self, src):
        direction = choice(DIRECTIONS)
        self.print(f"Moving on Random Direction: {direction}...")
        self.stop_cycle()
        
if __name__ == "__main__":
    for _ in range(NUM_WALKERS):
        Walker()
    Admin().start_system()