from maspy import *

class Sample(Agent):
    def __init__(self, name):
        super().__init__(name)
    
    @pl(gain,Goal("print"))
    def hello(self,src):
        self.print("Hello World")
        self.stop_cycle() # para terminar o agente
        
if __name__ == "__main__":
    print("A" == "a", 'a' == "a", "a" == 'A')
    exit()
    # agent_list = []  --> Nao necessario mais
    for i in range(10): # Reduzi para 10
        ag = Sample("Ag") # Nao necessario diferenciar o nome dos agentes 
        # ag = Sample(f"Ag{i}") # Mas pode ser assim tambem
        ag.connect_to(Channel("Private"))
        ag.add(Goal("print"))
        
    Admin().start_system() # Igual a 'for ag in agent_list: ag.reasoning_cycle()'
    
