from collections.abc import Iterable
from typing import Optional
from maspy.environment import Environment 
from maspy.agent import Agent, Belief, Goal, Plan
from maspy.admin import Admin

class Room(Environment):
    def __init__(self, env_name='room'):
        super().__init__(env_name)
        self.full_log = False
        self.create_percept("dirt",{(0,1): False, (2,2): False})

    def add_dirt(self, position):
        self.print(f"Dirt created in position {position}")
        dirt_status = self.get_percept_value("dirt")
        dirt_status.update({position: False}) 
        #self._update_fact("dirt",dirt_status) # same as below
    
    def clean_position(self, agent, position):
        self.print(f"{agent} is cleaning position {position}")
        dirt_status = self.get_percept_value("dirt")
        if dirt_status[position] is False:
            dirt_status[position] = True # changes the dict inside fact
        #self._update_fact("dirt",dirt_status) # useless cause of above

class Robot(Agent):
    def __init__(self, name, initial_env=None, full_log=False):
        super().__init__(name, full_log=full_log)
        self.connect_to(initial_env)
        self.add("o","decide_move")
        self.add("b","room_is_dirty")
        self.position = (0,0)
        self.print_beliefs
        self.print(f"Inicial position {self.position}")

    @Agent.plan("decide_move")
    def decide_move(self,src):
        min_dist = float("inf")
        target = None

        dirt_pos = self.get("b","dirt",1,"Room")
        print(f"{type(dirt_pos.args)}:{dirt_pos.args}")
        x, y = self.position
        for pos, clean in dirt_pos.args.items():
            if not clean:
                dist = abs(pos[0]-x) + abs(pos[1]-y)
                if dist < min_dist:
                    min_dist = dist
                    target = pos
                    
        if target is None:
            self.print(f"All dirt is cleaned")
            #self.rm_belief(Belief("room_is_dirty"))
            #self.add_belief(Belief("room_is_clean"))
            print("*** Finished Cleaning ***")
        else:
            self.print(f"Moving to {target}")
            self.add("o","move",(target,))
    
    @Agent.plan("clean")                            
    def clean(self,src):
        if self.has_belief(Belief("room_is_dirty")):
            self.execute_in("Room").clean_position(self.my_name, self.position)
            self.add(Goal("decide_move"))
    
    @Agent.plan("move")
    def move(self,src,target):
        x, y = self.position

        self.print(target," - ",x,y)
        if x != target[0]:
            diff = target[0] - x
            direction = (int(diff/abs(diff)),0)
        elif y != target[1]:
            diff = target[1] - y
            direction = (0,int(diff/abs(diff)))
        
        match direction:
            case (0,1): self.print(f"Moving Down")
            case (0,-1): self.print(f"Moving Up")
            case (-1,0): self.print(f"Moving Left")
            case (1,0): self.print(f"Moving Right")
        
        self.position = (x+direction[0],y+direction[1])
        self.print(f"New position: {self.position}")
        
        #print(f"{self.position} {target}")
        if self.position == target:
            self.print(f"Reached dirt position")
            self.add(Goal("clean"))
            return
        else:
            self.add("o","move",(target,))

def main(): 
    env = Room("Room")
    rbt = Robot('R1', initial_env=env, full_log=False)
    rbt.reasoning()
    env.add_dirt((3,1))
    rbt.add(Goal("decide_move"))

if __name__ == "__main__":
    main()

# Diagrama de classes
# Explicar Caracteristicas de agentes, ambiente, comunicação e controle do sistema

# Criação de N agentes
#   -> Crenças (Chave : Palavra, Argumentos: Qualquer Estrutura, Fonte: Palavra)
#   -> Objetivos (Chave : Palavra, Argumentos: Qualquer Estrutura, Fonte: Palavra)
#   -> Planos 
#   -> Percepção de N ambientes focados
#   -> Comunicação com N canais

# Criação de N ambientes
#   -> Fatos
#   -> Ações do ambiente

# Criação de N canais de comunicação
#   -> Descrição de canais 
#   -> lista de agentes conectados