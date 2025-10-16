from maspy import *
from random import choice, randint

DIRECTIONS = [
    "up", "down",
    "left", "right"
]
MAP_SIZE = (100, 100)
TARGET = randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1)

class Walker(Agent):
    def __init__(self, send_target=True):
        super().__init__()
        start_pos = (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))
        self.steps = 0
        self.send_target = send_target
        self.print(f'Start position: {start_pos}')
        self.add(Belief("my_pos", start_pos))
        self.add(Goal("move"))

    @pl(gain, Goal("move"), Belief("target", (Any,Any)) & Belief("my_pos", (Any,Any)))
    def make_move(self, src, tgt_x, tgt_y, pos_x, pos_y):
        #print(f'Knowing target {tgt_x, tgt_y} and my position {pos_x, pos_y}')
        self.steps += 1
        if pos_x != tgt_x:
            diff = tgt_x - pos_x
            direction = (int(diff/abs(diff)),0)
            if direction == (1,0):
                direction = "up"
            elif direction == (-1,0):
                direction = "down"
            else:
                print("Error up/down")
        elif pos_y != tgt_y:
            diff = tgt_y - pos_y
            direction = (0,int(diff/abs(diff)))
            if direction == (0,1):
                direction = "right"
            elif direction == (0,-1):
                direction = "left"
            else:
                print("Error left/right")
        else:
            self.print(f"Arrived in {self.steps} steps")
            self.stop_cycle()
            return
        
        my_pos = (pos_x, pos_y)
        
        new_pos = moviment(my_pos, direction)
        #self.print(f"Moving {direction} from {my_pos} to {new_pos}")
        self.rm(Belief("my_pos", my_pos))
        self.add(Belief("my_pos", new_pos))
        self.add(Goal("move"))
    
    @pl(gain, Goal("move"), ~Belief("target", (Any,Any)) & Belief("my_pos", (Any,Any)))
    def decide_move(self, src, pos_x, pos_y):
        my_pos = (pos_x, pos_y)
        direction = choice(DIRECTIONS)
        
        new_pos = moviment(my_pos, direction)
        #self.print(f"Randomly Moving {direction} from {my_pos} to {new_pos}")
        self.rm(Belief("my_pos", my_pos))
        self.add(Belief("my_pos", new_pos))
        
        self.steps += 1    
        if new_pos == TARGET:
            if self.send_target:
                list_walkers = self.list_agents("Walker")
                self.print(f"Arrived in {self.steps} steps, sending target to other Walkers")
                for walker in list_walkers:
                    if walker == self.my_name:
                        continue
                    self.send(walker, tell, Belief("target", new_pos))

            else:
                self.print(f"Arrived in {self.steps} steps")
            self.stop_cycle()
        else:
            self.add(Goal("move"))

def moviment(position: tuple[int, int], direction: str, step = 1) -> tuple[int, int]:
    moves = {
        "up": (step, 0),"down": (-step, 0), 
        "right": (0, step), "left": (0, -step),
    }
    new_x = max(0, min(position[0] + moves.get(direction, (0, 0))[0], MAP_SIZE[0] -1))  
    new_y = max(0, min(position[1] + moves.get(direction, (0, 0))[1], MAP_SIZE[1] -1))
    return (new_x, new_y)
        
if __name__ == "__main__":
    #Admin().set_logging(True,set_agents=False,set_admin=False,set_environments=False)
    Admin().print(f'Map size: {MAP_SIZE} - Target: {TARGET}')
    for _ in range(500):
        walker = Walker(send_target=True)
    Admin().start_system()