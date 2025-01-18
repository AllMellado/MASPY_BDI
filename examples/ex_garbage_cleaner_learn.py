# ruff: noqa
# mypy: ignore-errors

from maspy import *
from random import randint
import traceback

class Room(Environment):
    def __init__(self, env_name=None, full_log=False):
        super().__init__(env_name, full_log)
        self.create(Percept("room_is_dirty",adds_event=False))
    
    def clean_position(self, agent, position):
        self.print(f"{agent} is cleaning position {position}")
        dirt_status = self.get(Percept("dirt","Statuses"))
        assert isinstance(dirt_status.args,dict)
        if dirt_status.args[position] is False:
            dirt_status.args[position] = True 
        try:
            for status in dirt_status.args.values():
                if status is False:
                    break
            else:
                self.delete(Percept("room_is_dirty",adds_event=False))
        except Exception as ex:
            traceback.print_exc()

class Robot(Agent):
    def __init__(self, name, initial_env=None, full_log=False, print_path=True):
        super().__init__(name, show_exec=full_log)
        self.connect_to(initial_env)
        self.add(Goal("decide_move"))
        self.position = (0,0)
        self.charge_pos = (0,0)
        self.ml: Learning = None
        self.moves_made = 0
        self.print_path = print_path

    @pl(gain,Goal("decide_move"))
    def decide_move(self,src):
        if self.has(Belief("room_is_dirty",source="Room")):
            assert isinstance(self.ml,Learning)
            state = self.ml.env.encode(*self.position)
            action,n_state,reward = self.ml.exec(self.ml.env, state)
            self.add(Goal("move",(action,reward)))
        else:         
            self.print(f"All dirt is cleaned")
            print("*** Finished Cleaning ***")
            
            self.add(Goal("recharge"))
    
    @pl(gain,Goal("clean_dirt"),Belief("room_is_dirty"))                            
    def clean(self,src):
        self.action("Room").clean_position(self.tuple_name, self.position)
        self.print("Position cleaned. Deciding next move...") if self.print_path else ...
        self.add(Goal("decide_move"))
        
    @pl(gain,Goal("recharge"))   
    def return_charger(self,src):
        self.print("Return to charger")
        self.add(Goal("move_to",self.charge_pos))
    
    @pl(gain,Goal("move",("Action","Reward")))
    def move(self,src,action,reward):
        match action:
            case 0: 
                direction = (1,0) 
                self.print(f"Moving Down to ",end="") if self.print_path else ...
            case 1: 
                direction = (-1,0)
                self.print(f"Moving Up to ",end="") if self.print_path else ...
            case 2: 
                direction = (0,1)
                self.print(f"Moving Right to ",end="") if self.print_path else ...
            case 3: 
                direction = (0,-1)
                self.print(f"Moving Left to ",end="") if self.print_path else ...
        
        self.position = (self.position[0]+direction[0],
                         self.position[1]+direction[1])
        print(f"{self.position}") if self.print_path else ...
        
        self.moves_made += 1
        if reward > 0:
            self.print(f"Reached dirt position") if self.print_path else ...
            self.add(Goal("clean_dirt"))
        else:
            assert isinstance(self.ml,Learning)
            state = self.ml.env.encode(*self.position)
            action,n_state,reward = self.ml.exec(self.ml.env, state)
            #print(self.position," to ",list(self.ml.env.decode(n_state))," act: ",action," r[",reward,"]")
            self.add(Goal("move",(action,reward)))

    @pl(gain,Goal("move_to",("X","Y")))
    def move_to(self,src,tgX,tgY):
        x, y = self.position

        if x != tgX:
            diff = tgX - x
            direction = (int(diff/abs(diff)),0)
        elif y != tgY:
            diff = tgY - y
            direction = (0,int(diff/abs(diff)))
        else:
            self.print(f"Reached position. Recharging...")
            self.stop_cycle()
            return
        
        match direction:
            case (0,1): self.print(f"Moving Down to",end="") if self.print_path else ...
            case (0,-1): self.print(f"Moving Up to",end="") if self.print_path else ...
            case (-1,0): self.print(f"Moving Left to",end="") if self.print_path else ...
            case (1,0): self.print(f"Moving Right to",end="") if self.print_path else ...
        
        self.position = (x+direction[0],y+direction[1])
        print(f" {self.position}") if self.print_path else ...
        
        self.moves_made += 1
        if self.position == (tgX,tgY):
            self.print(f"Reached position. Recharging...")
            self.stop_cycle()
        else:
            self.add(Goal("move_to",(tgX,tgY)))

def main():    
    # Learning 
    lrn = Learning()
    lrn.set_params(map_size,num_targets,0,
                   num_pop,num_steps,num_iter)
    lrn.learn(method,show_log=True)
    
    # Environment Model
    map = [[0 for _ in range(map_size)] for _ in range(map_size)]
    dirt_percept = dict()
    for _ in range(num_targets):
        pos = (randint(0,map_size-1),randint(0,map_size-1))
        map[pos[0]][pos[1]] = -2
        dirt_percept[pos] = False
    lrn.set_env(map)
    
    # Environment
    env = Room()
    env.create(Percept("dirt",dirt_percept,adds_event=False))
    env.print(f'Dirt in positions: {dirt_percept.keys()}')
    
    # Agent
    rbt = Robot('R1', initial_env=env, full_log=False, print_path=True)
    rbt.charge_pos = (randint(0,map_size-1),randint(0,map_size-1))
    done = False
    while not done:
        rbt.position = (randint(0,map_size-1),randint(0,map_size-1))
        if rbt.position not in dirt_percept.keys(): done = True
    rbt.ml = lrn
    rbt.print(f"Inicial position {rbt.position} - Charging position {rbt.charge_pos}")
    
    # Admin
    Admin().start_system()
    print(f"Moves made by agent: {rbt.moves_made}")

if __name__ == "__main__":
    # Parameters 
    map_size = 5
    num_targets = 2
    method = 0

    epsilon = 0.1
    alpha = 0.1
    gamma = 0.9
    threshold = 0.001
    
    num_pop = 10
    num_steps = map_size*num_targets/2
    num_iter = 500
    
    main()