from maspy import *
from maspy.learning import *
from random import choice, randint
from threading import Lock
from itertools import product

NUM_WALKERS = 2
DIRECTIONS = [
    "up", "down",
    "left", "right"
]
MAP_SIZE = (20, 20)
steps: dict[str, int] = {}

COMBO = list(product(DIRECTIONS, repeat=NUM_WALKERS))
TARGET = randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1)

class Map(Environment):
    def __init__(self):
        self.pos_lock = Lock()
        super().__init__()
        self.moves = {
            "up": (1, 0),"down": (-1, 0), 
            "right": (0, 1), "left": (0, -1),
        }
        #self.create(Percept("target", MAP_SIZE, cartesian))
        self.create(Percept("target", TARGET))
        self.possible_starts = "off-policy"
    
    def on_connect(self, agt):
        start_pos = (randint(0, MAP_SIZE[0]-1), randint(0, MAP_SIZE[1]-1))
        #self.create(Percept("pos_agent", start_pos, agt))
        with self.pos_lock:
            post_learn = self.get(Percept("position", Any))
            if post_learn:
                #print(post_learn)
                post_learn.args[agt] = start_pos
                #self.change(post_learn, post_learn.args)
            else:
                self.create(Percept("position", {agt: start_pos}, listed))
        #post_learn = self.get(Percept("position", Any))
        #self.print(post_learn)
        self.print(f"Agent {agt} starts at {start_pos}")
    
    def move_transition(self, position_agents: dict[str, tuple], directions: dict[int, str]):
        # trivial reward and termination
        #print(" >",position_agents, flush=True)
        reward = -1
        terminated = False
        
        new_positions: dict = {}
        for agent, pos in position_agents.items():
            direction = directions[int(agent.split("_")[1])]
            
            dx, dy = self.moves.get(direction, (0, 0))
            raw_x = pos[0] + dx
            raw_y = pos[1] + dy

            if not (0 <= raw_x < MAP_SIZE[0] and 0 <= raw_y < MAP_SIZE[1]):
                reward -= 1

            new_x = max(0, min(raw_x, MAP_SIZE[0] - 1))
            new_y = max(0, min(raw_y, MAP_SIZE[1] - 1))
            
            #new_x = max(0, min(pos[0] + self.moves.get(direction, (0, 0))[0], MAP_SIZE[0] -1))  
            #new_y = max(0, min(pos[1] + self.moves.get(direction, (0, 0))[1], MAP_SIZE[1] -1))
            new_position = (new_x, new_y)
            new_positions[agent] = new_position
            
            # reward and termination when target is found by any agent
            if new_position == TARGET:
                reward = 10
                terminated = True
                
        #print(f'{position_agents} + {directions} \t > {new_positions} {terminated} {TARGET}')
        return new_positions, reward, terminated
    
    @action(listed, [{i+1: dir for i, dir in enumerate(combo)} for combo in COMBO], move_transition)
    def group_move(self, agt: str, directions: dict[int, str]):
        direction = directions[int(agt.split("_")[1])]
        self.move(agt, direction)
    
    def move(self, agt, direction):
        global steps
        steps[agt] += 1
        with self.pos_lock:
            #pos_agent = self.get(Percept("pos_agent", (Any, Any), agt), ck_group=True)
            #assert isinstance(pos_agent, Percept)
            pos_learn = self.get(Percept("position", Any))
            assert isinstance(pos_learn, Percept)
            #assert isinstance(pos_learn.args, dict)
            #pos_dict = pos_learn.args.copy()
            pos = pos_learn.args[agt]
            #assert pos == pos_agent.args, f"{agt} : {pos} != {pos_agent.args}"
            new_x = max(0, min(pos[0] + self.moves.get(direction, (0, 0))[0], MAP_SIZE[0] -1))  
            new_y = max(0, min(pos[1] + self.moves.get(direction, (0, 0))[1], MAP_SIZE[1] -1))
            new_pos = (new_x, new_y)
            pos_learn.args[agt] = new_pos
            #self.change(pos_agent, new_pos)
            
        #self.change(pos_learn, pos_dict)
        
        target = self.get(Percept("target", (Any, Any)))
        if isinstance(target, Percept) and new_pos == target.args:
            self.print(f"{agt} in {pos} moves {direction} and arrived at {target.args}")
            self.create(Percept("arrived_target", new_pos, agt))
        else:
            self.print(f"{agt} in {pos} moves {direction} to {new_pos}")

class Walker(Agent):
    def __init__(self, send_target=True):
        super().__init__()
        self.filter_perceptions(add, focus, ["listed", self.my_name])
        self.steps = 0
        self.send_target = send_target

    @pl(gain, Goal("move"), Belief("target", (Any,Any)) & Belief("position", Any, "Map"))
    def make_move(self, src, target, position):
        self.auto_action = False
        tgt_x, tgt_y = target
        pos_x, pos_y = position[0][self.my_name]
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
        self.print(f'Knowing target {tgt_x, tgt_y} and my position {pos_x, pos_y}  - moving {direction}')
        self.move(direction)
        #self.print(f"Moving {direction} from {my_pos} to {new_pos}")
        self.perceive("Map")
        #position = self.get(Belief("pos_agent", (Any,Any), "Map"))
        #self.print(f"Arrived at {position.args}")
        if not self.get(Belief("arrived_target", (Any,Any), "Map")):
            self.add(Goal("move"))
    
    @pl(gain, Belief("arrived_target", (Any,Any)), ~Belief("Guided"))
    def arrived_target(self, src, target):
        global steps
        self.print(f"Arrived at target - {steps[self.my_name]} steps")
        walkers = self.list_agents("Walker")
        print(f"Sending target to {walkers}")
        for walker in walkers:
            if walker == self.my_name:
                continue
            self.print(f"Sending target to {walker}")
            self.send(walker, tell, [Belief("target", target), Belief("Guided")])
            self.send(walker, achieve, Goal("move"))
        self.stop_cycle()
    
    @pl(gain, Belief("arrived_target", (Any,Any)))
    def guided_target(self, src, target):
        global steps
        self.print(f"Guided Arrival on target - {steps[self.my_name]} steps")
        print("All Steps Taken: \n",steps)
        self.stop_cycle()
        
if __name__ == "__main__":
    Admin().console_settings(True,set_agents=False,set_admin=False,set_environments=False)
    Admin().print(f'Map size: {MAP_SIZE} - Target: {TARGET}')
    map = Map()
    walkers: list[Walker] = []
    for i in range(NUM_WALKERS):
        walkers.append(Walker(send_target=True))
        steps[walkers[i].my_name] = 0
    Admin().connect_to(walkers, map)
    model = EnvModel(map)
    print(f'Actions: {model.actions_list}  Space: {model.observation_space}')
    #model.load_learning("Map_qlearning_0.05_0.8_1_0.1_10000_25.pkl")
    model.learn(qlearning, num_episodes=10000, max_steps=25, load_learning=False)
    #for i,j in model.q_table.items():
    #    print(f'{i}: {j}')
    model.reset_percepts()
    for walker in walkers:
        walker.add_policy(model)
        walker.auto_action = True
    #Admin().slow_cycle_by(0.5)

    target = map.get(Percept("target", (Any, Any)))
    if isinstance(target, Percept):
        target = target.args
    Admin().print(f'Map size: {MAP_SIZE} - Target: {target}')
    Admin().start_system()