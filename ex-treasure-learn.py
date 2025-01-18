from maspy import *
from maspy.learning import *

class TreasureChest(Environment):
    def __init__(self):
        super().__init__()
        self.create(Percept("chests", [(1,0,0),(0,1,0),(0,0,1)], listed))
        self.create(Percept("treasure", ("closed", "found", "trap"), listed))
        
    
    def open_chest_transition(self, state: dict, open: int):
        chests = state['chests']
        if chests[open] == 1:
            reward = 10
        else:
            reward = -5
        return state, reward, True