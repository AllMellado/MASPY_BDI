from maspy import *
from random import choice, randint

DIRECTIONS = [
    "up", "down",
    "left", "right"
]
MAP_SIZE = (5, 5)
TARGET = randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1)

class Walker(Agent):
    def __init__(self):
        super().__init__()
        self.add(Belief("my_pos", (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))))
        self.add(Goal("move"))
    
    @pl(gain, Goal("move"))
    def decide_move(self,src):
        direction = choice(DIRECTIONS)
        my_pos = self.get(Belief("my_pos", (Any,Any)))
        
        if isinstance(my_pos, Belief):
            new_pos = moviment(my_pos.args, direction)
            self.print(f"Moving {direction} from {my_pos.args} to {new_pos}")
            self.rm(my_pos)
            self.add(Belief("my_pos", new_pos))
            
        if new_pos == TARGET:
            self.print("Arrived")
            self.stop_cycle()
        else:
            self.add(Goal("move"))

def moviment(position: tuple[int, int], direction: str, step = 1) -> tuple[int, int]:
    moves = {
        "up": (0, step), "down": (0, -step),
        "left": (-step, 0), "right": (step, 0),
    }
    new_x = max(0, min(position[0] + moves.get(direction, (0, 0))[0], MAP_SIZE[0] -1))  
    new_y = max(0, min(position[1] + moves.get(direction, (0, 0))[1], MAP_SIZE[1] -1))
    return (new_x, new_y)
        
if __name__ == "__main__":
    Admin().print(f'Map size: {MAP_SIZE} - Target: {TARGET}')
    walker = Walker()
    Admin().start_system()