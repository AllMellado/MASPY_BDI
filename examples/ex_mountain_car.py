# ruff: noqa: F403, F405
from maspy import *
from maspy.learning.modelling import EnvModel
import numpy as np

class Mountain_Car(Environment):
    def __init__(self, env_name) -> None:
        super().__init__(env_name)
        
        self.create(Percept("Car_position_x_axis", [-1.2,0.6], group="interval"))
        self.create(Percept("Car_velocity", [-0.07,0.07], group="interval"))
    
    @action("interval", [-1, 1])    
    def apply_force(self, state, force):
        reward = 0
        
        return state, reward

if __name__ == "__main__":
    car = Mountain_Car('Car1')
    model = EnvModel(car)
