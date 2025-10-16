from maspy import *
from random import choice, randint

NUM_WALKERS = 10
DIRECTIONS = [
    "up", "down",
    "left", "right"
]
MAP_SIZE = (10, 10)
steps: dict[str, int] = {}


class Map(Environment):
    def __init__(self):
        super().__init__()
        self.moves = {
            "up": (1, 0),"down": (-1, 0), 
            "right": (0, 1), "left": (0, -1),
        }
        target = (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))
        self.print(f"Target is at {target}")
        self.create(Percept("target", target))
        
    def on_connect(self, agt):
        start_pos = (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))
        self.create(Percept("agt_position", start_pos, agt))
        self.print(f"Agent {agt} starts at {start_pos}")
    
    def moviment(self, position, direction):
        dx, dy = self.moves[direction]
        new_x = max(0, min(position[0] + dx, MAP_SIZE[0] - 1))
        new_y = max(0, min(position[1] + dy, MAP_SIZE[1] - 1)) 
        return new_x, new_y
    
    def move(self, agt, direction):
        position = self.get(Percept("agt_position", (Any, Any), agt), ck_group=True)
        assert isinstance(position, Percept)
        
        pos = position.args
        if direction == "stay":
            new_pos = pos
        else:
            new_pos = self.moviment(pos, direction)
            self.change(position, new_pos)
        
        global steps
        steps[agt] += 1
        target = self.get(Percept("target", (Any, Any)))
        if isinstance(target, Percept) and new_pos == target.args:
            self.print(f"{agt} in {pos} moves {direction} and arrived at {target.args}")
            self.create(Percept("arrived_target", new_pos, agt))
        else:
            self.print(f"{agt} in {pos} moves {direction} to {new_pos}")
    
class Walker(Agent):
    def __init__(self):
        super().__init__()
        global steps
        steps[self.my_name] = 0
        self.filter_perceptions(add, focus, self.my_name)
        self.add(Goal("move"))

    @pl(gain, Goal("move"), Belief("target", (Any,Any)) & Belief("agt_position", (Any,Any)))
    def best_move(self, src, target, position):
        dx, dy = target[0] - position[0], target[1] - position[1]
        if dx != 0:
            direction = "up" if dx > 0 else "down"
        elif dy != 0:
            direction = "right" if dy > 0 else "left"
        else:
            direction = "stay"
            
        self.print(f'Knowing target {target} and my position {position}  - moving {direction}')
        self.move(direction)
        
        self.perceive("Map")
        if not self.has(Belief("arrived_target", (Any,Any), "Map")):
            return False
        else:
            self.stop_cycle()
            #print(f"Driver {self.my_name} arrived at {target}")
            
    
    @pl(gain, Goal("move"))
    def make_move(self, src):
        direction = choice(DIRECTIONS)
        self.print(f"Moving on Random Direction: {direction}...")
        self.move(direction)
        
        self.perceive('Map')
        
        if not self.has(Belief("arrived_target", (Any,Any), "Map")):
            return False
        else:
            target = self.get(Belief("arrived_target", (Any,Any), "Map"))
            walkers = self.list_agents("Walker")
            print(f"Driver {self.my_name} arrived at {target} first, sending target")
            self.send(broadcast, Belief("target", target.args))
            print(f"Driver {self.my_name} finished broadcast")
            self.stop_cycle()

def main(num_walkers: int):
    Admin().block_prints()
    map = Map()
    walkers = []
    for _ in range(num_walkers):
        walkers.append(Walker())
    Admin().connect_to(walkers, map)
    Admin().start_system()
    print(steps)
    print(f'Elapsed time: {Admin().elapsed_time} seconds - {Admin().elapsed_time/60} minutes')

if __name__ == "__main__":
    #for num_walkers in [1,5,10,50,100,500,1000,5000]:
    #    Admin().print(f"Executing System with {num_walkers} walkers")
    main(100)
    Admin().reset_instance()