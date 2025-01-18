from maspy import *
from maspy.learning import *

class Map(Environment):
    def __init__(self, env_name=None):
        super().__init__(env_name)
        
        self.targets = [(1,4),(3,2)]
        self.max_row = 5
        self.max_col = 5
        size = (self.max_row+1,self.max_col+1)
        self.create(Percept("location", size, cartesian))
        self.possible_starts = {"location": [(0,0),(4,4),(0,5),(5,3)]}
    
    def move_transition(self, state: dict, direction: str):
        location = state["location"]
        
        location =self.moviment(location, direction)
            
        state["location"] = location
        
        if location in self.targets:
            reward = 10
            terminated = True
        else:
            reward = -1
            terminated = False
        return state, reward, terminated

    @action(listed, ("up","down","left","right"), move_transition)
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
    
class Sample(Agent):
    pass
        
if __name__ == "__main__":
    env = Map()
    model = EnvModel(env)
    print(f'actions: {model.action_space}  space: {model.observation_space}')
    model.learn(qlearning)
    ag = Sample()
    ag.add_policy(model)
    Admin().start_system()
    