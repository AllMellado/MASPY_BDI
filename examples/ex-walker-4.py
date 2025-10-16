from maspy import *
from random import choice, randint

DIRECTIONS = [
    "up", "down",
    "left", "right"
]
MAP_SIZE = (20, 20)
TARGET = randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1)

class Map(Environment):
    def __init__(self):
        super().__init__()
        self.moves = {
            "up": (1, 0),"down": (-1, 0), 
            "right": (0, 1), "left": (0, -1),
        }
        self.create(Percept("target", TARGET))
    
    def on_connect(self, agt):
        start_pos = (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))
        self.create(Percept("position", start_pos, agt))
        self.print(f"Agent {agt} starts at {start_pos}")
    
    def check_target(self, agt):
        position = self.get(Percept("position", (Any, Any), agt), ck_group=True)
        target = self.get(Percept("target", (Any, Any)))
        return True if position.args == target.args else False
    
    def move(self, agt, direction):
        position = self.get(Percept("position", (Any, Any), agt), ck_group=True)
        if position:
            #self.print(f"Agent:{agt} on position {position.args} moves {direction}")
            pos = position.args
        else:
            self.print(f"Agent:{agt} has no position set")
            return
        new_x = max(0, min(pos[0] + self.moves.get(direction, (0, 0))[0], MAP_SIZE[0] -1))  
        new_y = max(0, min(pos[1] + self.moves.get(direction, (0, 0))[1], MAP_SIZE[1] -1))
        self.change(position, (new_x, new_y))
        
class Walker(Agent):
    def __init__(self, send_target=True):
        super().__init__()
        self.filter_perceptions(add, focus, self.my_name)
        self.steps = 0
        self.send_target = send_target
        self.add(Goal("move"))

    @pl(gain, Goal("move"), Belief("target", (Any,Any)) & Belief("position", (Any, Any), "Map"))
    def make_move(self, src, target, position):
        tgt_x, tgt_y = target
        pos_x, pos_y = position
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
        #print(f'Knowing target {tgt_x, tgt_y} and my position {pos_x, pos_y}  - moving {direction}')
        self.move(direction)
        #self.print(f"Moving {direction} from {my_pos} to {new_pos}")
        self.add(Goal("move"))
    
    @pl(gain, Goal("move"), ~Belief("target", (Any,Any)))
    def decide_move(self, src):
        direction = choice(DIRECTIONS)
        #self.print_beliefs
        #self.print(f"Randomly Moving {direction}")
        self.move(direction)
        
        self.steps += 1    
        if self.check_target():
            if self.send_target:
                list_walkers = self.list_agents("Walker")
                self.print(f"Arrived in {self.steps} steps, sending target to other Walkers")
                target = self.get(Belief("position", (Any, Any), "Map"))
                for walker in list_walkers:
                    if walker == self.my_name:
                        continue
                    self.send(walker, tell, Belief("target", target.args))

            else:
                self.print(f"Arrived in {self.steps} steps")
            self.stop_cycle()
        else:
            self.add(Goal("move"))
        
if __name__ == "__main__":
    #Admin().set_logging(True,set_agents=False,set_admin=False,set_environments=False)
    Admin().print(f'Map size: {MAP_SIZE} - Target: {TARGET}')
    map = Map()
    walkers = []
    for _ in range(5):
        walkers.append(Walker(send_target=True))
    Admin().connect_to(walkers, map)
    #Admin().slow_cycle_by(0.5)
    Admin().start_system()