from maspy import *
from maspy.learning import *
from random import randint
import pygame 
import sys 
from time import sleep
from threading import Thread
from collections import deque
import random
import os

os.environ['SDL_VIDEO_CENTERED'] = '1'

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
SIDEBAR_WIDTH = 200
AGENT_COLOR = (0, 100, 255)
TARGET_COLOR = (255, 50, 50)
BG_COLOR = (240, 240, 240)
GRID_COLOR = (200, 200, 200)

NUM_WALKERS = 10
DIRECTIONS = [
    "up", "down",
    "left", "right"
]
steps: dict[str, int] = {}

BUTTON_COLOR = (80, 80, 200)
BUTTON_HOVER = (100, 100, 250)
TEXT_COLOR = (255, 255, 255)
SIDEBAR_WIDTH = 200

pygame.init()
FONT = pygame.font.SysFont(None, 24)

class Button:
    def __init__(self, rect, text, callback):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.hover = False

    def draw(self, screen):
        color = BUTTON_HOVER if self.hover else BUTTON_COLOR
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        text_surf = FONT.render(self.text, True, TEXT_COLOR)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.hover:
            self.callback()

class TextInput:
    def __init__(self, rect, label, default=""):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.text = default
        self.active = False

    def draw(self, screen):
        pygame.draw.rect(screen, (70, 70, 70), self.rect, border_radius=5)
        if self.active:
            pygame.draw.rect(screen, (150, 150, 250), self.rect, 2)
        else:
            pygame.draw.rect(screen, (120, 120, 120), self.rect, 2)
        txt_surface = FONT.render(self.text, True, TEXT_COLOR)
        label_surface = FONT.render(self.label, True, TEXT_COLOR)
        screen.blit(label_surface, (self.rect.x - 120, self.rect.y + 5))
        screen.blit(txt_surface, (self.rect.x + 10, self.rect.y + 5))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode

class MapVisualizer:
    def __init__(self, env) -> None:
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

        # Compute cell size to fit inside remaining space
        self.map_area_width = WINDOW_WIDTH - SIDEBAR_WIDTH
        self.map_area_height = WINDOW_HEIGHT
        self.cell_size = min(
            self.map_area_width // env.map_size[0],
            self.map_area_height // env.map_size[1],
        )

        #width = env.map_size[0]*self.cell_size + SIDEBAR_WIDTH
        #height = env.map_size[1]*self.cell_size
        #self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Map Environment")
        self.clock = pygame.time.Clock()
        self.buttons: list[Button] = self._create_buttons()
    
    def _create_buttons(self):
        buttons = []
        x = self.env.map_size[0]*self.cell_size + 20
        y = 40
        spacing = 60
        actions = [self.action_1, self.action_2, self.action_3, self.action_4]
        names = ["Pause/Resume", "More Delay", "Less Delay", "Reset"]
        for i, func in enumerate(actions):
            btn = Button(
                rect=(x, y + i*spacing, SIDEBAR_WIDTH-40, 40),
                text= names[i],
                callback=func
            )
            buttons.append(btn)
        return buttons

    # --- Placeholders for later logic ---
    def action_1(self): Admin().pause_system()
    def action_2(self): Admin().slower_cycle()
    def action_3(self): Admin().faster_cycle()
    def action_4(self): Admin()._models[0].reset_percepts()
    def action_5(self): print("Action 5 triggered")
    def action_6(self): print("Action 6 triggered")
    
    def draw_sidebar(self):
        sidebar_rect = pygame.Rect(self.env.map_size[0]*self.cell_size, 0, SIDEBAR_WIDTH, self.env.map_size[1]*self.cell_size)
        pygame.draw.rect(self.screen, (30, 30, 60), sidebar_rect)
        for btn in self.buttons:
            btn.draw(self.screen)
    
    def draw_map(self):
        # grid
        for x in range(0, self.env.map_size[0]*self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.env.map_size[1]*self.cell_size))
        for y in range(0, self.env.map_size[1]*self.cell_size, self.cell_size):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self.env.map_size[0]*self.cell_size, y))

        # draw target
        target = self.env.get(Percept("target", (Any, Any)))
        if target:
            tx, ty = target.args
            cx, cy = tx * self.cell_size, ty * self.cell_size
            pygame.draw.line(self.screen, TARGET_COLOR,
                            (cx + 4, cy + 4),
                            (cx + self.cell_size - 4, cy + self.cell_size - 4), 3)
            pygame.draw.line(self.screen, TARGET_COLOR,
                            (cx + self.cell_size - 4, cy + 4),
                            (cx + 4, cy + self.cell_size - 4), 3)

        # draw agents
        for percept in self.env.get(Percept("agt_position", (Any, Any)), all=True):
            x, y = percept.args
            cx = x * self.cell_size + self.cell_size // 2
            cy = y * self.cell_size + self.cell_size // 2
            body = self.cell_size // 4

            # body
            pygame.draw.rect(self.screen, AGENT_COLOR,
                             (cx - body//2, cy - body//2, body, body))

            # spider legs: 4 angled lines downward
            leg_len = self.cell_size // 3
            # body bottom corners/sides
            bottom_left  = (cx - body//2, cy + body//2)
            bottom_right = (cx + body//2, cy + body//2)
            bottom_mid   = (cx,          cy + body//2)

            # left leg (angled left-down)
            pygame.draw.line(self.screen, AGENT_COLOR,
                            bottom_left,
                            (bottom_left[0] - leg_len//2, bottom_left[1] + leg_len), 2)

            # left-mid leg (straight down)
            pygame.draw.line(self.screen, AGENT_COLOR,
                            (cx - body//4, cy + body//2),
                            (cx - body//4, cy + body//2 + leg_len), 2)

            # right-mid leg (straight down)
            pygame.draw.line(self.screen, AGENT_COLOR,
                            (cx + body//4, cy + body//2),
                            (cx + body//4, cy + body//2 + leg_len), 2)

            # right leg (angled right-down)
            pygame.draw.line(self.screen, AGENT_COLOR,
                            bottom_right,
                            (bottom_right[0] + leg_len//2, bottom_right[1] + leg_len), 2)

    def draw(self):
        self.screen.fill(BG_COLOR)
        self.draw_map()
        self.draw_sidebar()
        pygame.display.flip()
        
    def loop(self):
        running = True
        while running:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    for btn in self.buttons:
                        btn.handle_event(event)
            except Exception as e:
                print(e)
                sys.exit()
            
            self.draw()
            
            pygame.time.delay(10)
            self.clock.tick(30)  # FPS
            
        pygame.quit()
        
class MenuScreen:
    def __init__(self):
        self.screen = pygame.display.set_mode((500, 400))
        pygame.display.set_caption("Simulation Setup")
        self.clock = pygame.time.Clock()
        self.inputs = [
            TextInput((200, 100, 150, 35), "Map Width:", "20"),
            TextInput((200, 160, 150, 35), "Map Height:", "20"),
            TextInput((200, 220, 150, 35), "Agents:", "5")
        ]
        self.start_button = Button((175, 300, 150, 50), "Start Simulation", self.start)
        self.done = False
        self.settings = None

    def start(self):
        try:
            width = int(self.inputs[0].text)
            height = int(self.inputs[1].text)
            agents = int(self.inputs[2].text)
            self.settings = ((width, height), agents)
            self.done = True
        except ValueError:
            print("Invalid input")

    def loop(self):
        while not self.done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                self.start_button.handle_event(event)
                for inp in self.inputs:
                    inp.handle_event(event)

            self.screen.fill((20, 20, 50))
            for inp in self.inputs:
                inp.draw(self.screen)
            self.start_button.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(30)
        return self.settings

def generate_map_with_walls(width, height, target, wall_prob=0.2, max_attempts=1000):
    def neighbors(x, y):
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < width and 0 <= ny < height:
                yield nx, ny

    for _ in range(max_attempts):
        # 1. Random walls (excluding target)
        walls = {(x, y) for x in range(width) for y in range(height)
                 if random.random() < wall_prob and (x, y) != target}

        # 2. Flood-fill from target to find reachable open cells
        queue = deque([target])
        reachable = {target}

        while queue:
            cx, cy = queue.popleft()
            for nx, ny in neighbors(cx, cy):
                if (nx, ny) not in walls and (nx, ny) not in reachable:
                    reachable.add((nx, ny))
                    queue.append((nx, ny))

        # 3. Check if there are no isolated open areas
        total_open = width * height - len(walls)
        if len(reachable) == total_open:
            available = sorted(reachable)
            return walls, available

    raise RuntimeError("Failed to generate a valid map after many attempts.")

class Map(Environment):
    def __init__(self, map_size):
        super().__init__()
        self.moves = {
            "up": (-1, 0),"down": (1, 0), 
            "right": (0, 1), "left": (0, -1),
        }
        self.map_size = map_size
        target = (randint(0, self.map_size[0]-1), randint(0, self.map_size[1]-1))
        self.print(f"Target is at {target}")
        #positions = generate_map_with_walls(map_size[0], map_size[1], target)
        self.create(Percept("target", target))
        self.create(Percept("position", self.map_size, cartesian))
        self.possible_starts = "off-policy"
        
    def on_connect(self, agt):
        start_pos = (randint(0, self.map_size[0]-1), randint(0, self.map_size[1]-1))
        self.create(Percept("agt_position", start_pos, agt))
        self.print(f"Agent {agt} starts at {start_pos}")
    
    def moviment(self, position, direction):
        dx, dy = self.moves[direction]
        new_x = max(0, min(position[0] + dx, self.map_size[0] - 1))
        new_y = max(0, min(position[1] + dy, self.map_size[1] - 1)) 
        return new_x, new_y
    
    def move_transition(self, position, direction):
        reward = -1
        terminated = False
        
        new_position = self.moviment(position, direction)
        target = self.get(Percept("target", (Any, Any)))
        if new_position == target.args:
            reward = 10
            terminated = True
        return new_position, reward, terminated
    
    @action(listed, DIRECTIONS, move_transition)
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
    def __init__(self, name, map: Environment):
        super().__init__(name, max_intentions=1)
        global steps
        steps[self.my_name] = 0
        #self.connect_to(map)
        model = EnvModel(map, self)
        print(model)
        model.learn(qlearning, num_episodes=1000, max_steps=25)
        #for i,j in model.q_table.items():
        #    print(f'{i}: {j}')
        self.add_policy(model)
        self.filter_perceptions(add, focus, self.my_name)
        self.add(Goal("move"))

    @pl(gain, Goal("move"), Belief("target", (Any,Any)) & Belief("agt_position", (Any,Any), "Map"))
    def best_move(self, src, target, position):
        dx, dy = target[0] - position[0], target[1] - position[1]
        if dx == 0 and dy == 0:
            direction = "stay"
        elif abs(dx) >= abs(dy):
            direction = "down" if dx > 0 else "up"
        else:
            direction = "right" if dy > 0 else "left"
            
        self.print(f'Knowing target {target} and my position {position}  - moving {direction}')
        if direction != "stay": self.move(direction)
        
        self.perceive("Map")
        if not self.has(Belief("arrived_target", (Any,Any), "Map")):
            return False
        else:
            self.stop_cycle()
            #print(f"Driver {self.my_name} arrived at {target}")
            
    @pl(gain, Goal("move"), Belief("agt_position", (Any,Any), "Map"))
    def make_move(self, src, position):
        self.print(f"From {position} Moving on a Learned Direction...")
        self.best_action("Map",(position,))
        self.perceive("Map")
        
        if not self.has(Belief("arrived_target", (Any,Any), "Map")):
            return False # Retries current intention 
        else:
            target = self.get(Belief("arrived_target", (Any,Any), "Map"))
            walkers = self.list_agents("Walker")
            print(f"Driver {self.my_name} arrived at {target} first, sending target")
            self.sendf(broadcast, Belief("target", target.args))
            print(f"Driver {self.my_name} finished broadcast")
            self.stop_cycle()

def main():
    #Admin(console_log=True)
    #Channel().show_exec = True
    menu = MenuScreen()
    settings = menu.loop()
    if settings:
        map_size, num_walkers = settings
        map = Map(map_size)
        walkers = []
        for _ in range(num_walkers):
            wlk = Walker("wk", map)
            walkers.append(wlk)
        #Admin().connect_to(walkers, map)
        #Admin().slow_cycle_by(1)
        
        Admin().slow_cycle_by(.2)
        Thread(target=Admin().start_system, daemon=True).start()
        vis = MapVisualizer(map)
        vis.loop()
        #sleep(.5)
        #pygame.quit()
        print(steps)
        print(f'Elapsed time: {Admin().elapsed_time} seconds - {Admin().elapsed_time/60} minutes')
        #Admin().reset_instance()


if __name__ == "__main__":
    #for num_walkers in [1,5,10,50,100,500,1000,5000]:
    #    Admin().print(f"Executing System with {num_walkers} walkers")
    main()
    #Admin().reset_instance()