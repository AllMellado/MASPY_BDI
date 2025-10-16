from maspy import *
from maspy.learning import *

import random
from itertools import product


WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20

class SnakeEnv(Environment):
    def __init__(self, env_name="SnakeEnv"):
        super().__init__(env_name)
        self.w = 640
        self.h = 480
        self.max_col = self.w // BLOCK_SIZE
        self.max_row = self.h // BLOCK_SIZE
        self.score = 0
        # Percepts
        self.create(Percept("head_location", (self.max_col, self.max_row), cartesian))
        self.create(Percept("food_location", (self.max_col, self.max_row), cartesian))
        self.create(Percept("snake_body", [((10, 14),)], listed)) # Change to list of tuples
        # "head_location": [(x, y) for x in range(self.max_col) for y in range(self.max_row)],
        # "food_location": [(x, y) for x in range(self.max_col) for y in range(self.max_row)],
        # "snake_body": [((x, y),) for x in range(self.max_col) for y in range(self.max_row)]
        self.possible_starts = {
            "head_location": [(10, 15)],
            "food_location": [(16, 12)],
            "snake_body": [((10, 14),)] # Also Change it here for the possible starts
        }
        #self.reset()
    
    def food_percept(self):
        return Percept("food_location", (self.max_col, self.max_row), cartesian)
    def head_percept(self):
        return Percept("head_location", (self.max_col, self.max_row), cartesian)
    def body_percept(self):
        return Percept("snake_body", [(10, 14)], listed)
    
    def reset(self):
        # Cabeça em (10, 15), corpo em (10, 14) (tamanho 2)
        percept = self.get(Percept("head_location", (Any, Any)))
        self.change(percept, (10, 15))
        percept = self.get(Percept("food_location", (Any, Any)))
        self.change(percept, (16, 12))
        percept = self.get(Percept("snake_body", Any))
        self.change(percept, [(10, 14)])
        self.score = 0
        self.frame_iteration = 0
    
    def place_food(self, state):
        while True:
            new_food = (random.randint(0, self.max_col - 1), random.randint(0, self.max_row - 1))
            if new_food not in state["snake_body"]:
                break
        percept = self.get(Percept("food_location", (Any, Any)))
        self.change(percept, new_food)
    
    def state_to_tuple(self, state):
        # Garante a ordem dos percepts igual ao possible_starts
        if isinstance(state, dict):
            return (
                state["head_location"],
                state["food_location"],
                state["snake_body"]
            )
        return state
    
    def get_current_state(self):
        return (
            self.get(Percept("head_location", (Any, Any))).args,
            self.get(Percept("food_location", (Any, Any))).args,
            self.get(Percept("snake_body", Any))
        )
    
    def play_step(self, state, action):
        state = self.state_to_tuple(state)
        reward = 0
        self.frame_iteration += 1

        new_state, reward, done = self.move_transition(state, action)
        if not done:
            self.apply_action(new_state)
        else:
            self.reset()
            return reward, done, self.score
        self._update_ui(new_state)
        self.clock.tick(SPEED)
        return new_state, reward, done
    
    def apply_action(self, new_state):
        # Aceita tupla ou dicionário
        if isinstance(new_state, tuple):
            new_state = {
                "head_location": new_state[0],
                "food_location": new_state[1],
                "snake_body": new_state[2]
            }
        percept = self.get(Percept("head_location", (Any, Any)))
        self.change(percept, new_state["head_location"])
        percept = self.get(Percept("food_location", (Any, Any)))
        self.change(percept, new_state["food_location"])
        percept = self.get(Percept("snake_body", Any))
        self.change(percept, new_state["snake_body"])
    
    def move_transition(self, state, direction):
        #print("move_transition state:", state)
        state = self.state_to_tuple(state)
        head = state[0]
        food = state[1]
        body = state[2]
        
        if isinstance(body, tuple):
            body = [body]
        else:
            body = list(body)
        new_head = self.calculate_new_position(head, direction)
        body.insert(0, head)
        reward, done = -0.1, False
        if self.is_collision(new_head, body):
            body = tuple(body)
            #print(f'\tReturning "head_location": {new_head}, "food_location": {food}, "snake_body": {body}, -10, True')
            return {"head_location": new_head, "food_location": food, "snake_body": body}, -10, True
        elif new_head == food:
            reward = 10
            self.score += 1
            self.place_food({"head_location": new_head, "food_location": food, "snake_body": body})
        else:
            body.pop()
        body = tuple(body)
        if reward == 10:
            print(f'\tReturning "head_location": {new_head}, "food_location": {food}, "snake_body": {body}, {reward}, {done}')
        return {"head_location": new_head, "food_location": food, "snake_body": body}, reward, done
    
    def calculate_new_position(self, head, direction):
        x, y = head
        if direction == "UP":
            return (x, y - 1)
        elif direction == "DOWN":
            return (x, y + 1)
        elif direction == "LEFT":
            return (x - 1, y)
        elif direction == "RIGHT":
            return (x + 1, y)
        return head
    
    def is_collision(self, head, body):
        return (
            head[0] < 0 or head[0] >= self.max_col or
            head[1] < 0 or head[1] >= self.max_row or
            head in body
        )
    
    def _update_ui(self, state):
        if isinstance(state, tuple):
            state = {
                "head_location": state[0],
                "food_location": state[1],
                "snake_body": state[2]
            }

    
    @action(listed, ("UP", "DOWN", "LEFT", "RIGHT"), move_transition)
    def move(self, agt, direction):
        state = self.get_current_state()
        new_state, reward, done = self.move_transition(state, direction)
        if not done:
            self.apply_action(new_state)
        return reward, done

class SnakeAgent(Agent):
    pass

if __name__ == "__main__":
    env = SnakeEnv()
    model = EnvModel(env)
    print(f'actions: {model.actions_list}  space: {model.observation_space}')
    model.learn(qlearning)
    SnakeAgent("Ag").add(Goal("aquire_learning", [model]))
    Admin().start_system()