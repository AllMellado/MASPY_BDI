from maspy import *
from maspy.learning import *
from random import choice

IDS = [0, 1, 2, 3]

class Map(Environment):
    def __init__(self, env_name=None):
        super().__init__(env_name)

        self.targets = [(1,4),(3,2)]
        self.max_row = 5
        self.max_col = 5
        size = (self.max_row+1,self.max_col+1)
        num_agentes = len(IDS)
        for i in range(num_agentes):
            self.create(Percept(f"location_ag_{i}", size, cartesian))
        self.possible_starts = {"location": [(0,0),(4,4),(0,5),(5,3)]}
    
    def move_transition(self, state: dict, id_direction: tuple[int, str]):
        id, direction = id_direction
        location = state[f"location_ag_{id}"]
        
        location =self.moviment(location, direction)

        state["location"] = location
        
        if location in self.targets:
            reward = 10
            terminated = True
        else:
            reward = -1
            terminated = False
        return state, reward, terminated

    @action(cartesian, (IDS,  ("up","down","left","right")), move_transition)
    def move(self, agt, direction: str):
        percept = self.get(Percept("location", (Any,Any)))
        assert isinstance(percept, Percept)
        self.print(f"{agt} in {percept.args} is Moving {direction}")
        new_location = self.moviment(percept.args, direction)
        self.change(percept, new_location)
        self.print_percepts
        if new_location in self.targets:
            self.print(f"{agt} reached a target")
    
    def moviment(self, location, direction):
        if direction == "up" and location[0] > 0:
            location = (location[0]-1, location[1])
        elif direction == "down" and location[0] < self.max_row:
            location = (location[0]+1, location[1])
        elif direction == "left" and location[1] > 0:
            location = (location[0], location[1]-1)
        elif direction == "right" and location[1] < self.max_col:
            location = (location[0], location[1]+1)
        return location
    
class Mover(Agent):
    def __init__(self, my_name=None):
        super().__init__(my_name)
        self.add(Goal("ask_for_guide"))

    @pl(gain,Goal("ask_for_guide"))
    def move_guide(self,src) -> None:
        self.print("Asking for guide")
        belief_model = self.send("Instructor", askOneReply, Belief("Model",("Map", Any)))
        if isinstance(belief_model, Belief):
            model: EnvModel = belief_model.args[1][0]
            model.reset_percepts()
            self.add_policy(model)
        else:
            self.print("No model available for Map")
            self.wait(3)
            self.add(Goal("ask_for_guide"))

class Instructor(Agent):
    def __init__(self, my_name=None, env=None):
        super().__init__(my_name)
        self.add(Goal("make_model",[env]))
    
    @pl(gain,Goal("make_model",Any))
    def make_model(self,src,env: list[Environment]):
        self.print(f"Making model for {env[0].my_name}")
        model = EnvModel(env[0])
        print(f'actions: {model.action_space}  space: {model.observation_space}')
        model.learn(qlearning)
        self.print(f"Finished training {env[0].my_name} Model")
        self.add(Belief("Model",(env[0].my_name, [model])))
            
if __name__ == "__main__":
    mv = Mover()
    map = Map()
    it = Instructor(env=map)
    Admin().start_system()
