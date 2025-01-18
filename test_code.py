
# ruff: noqa: F403, F405
from maspy import *
from maspy.learning import *
from maspy.learning.modelling import Group

class Env(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        
    def func_env(self, agt):
        self.print(f"Hello to agent {agt}")

class Ag(Agent):
    def __init__(self, ag_name):
        super().__init__(ag_name)
    
    def test3(self):
        self.wait(event=Event(gain, Belief("A")))
    
    def test2(self):
        self.test3()
        #self.wait(event=Event(gain, Belief("A")))
    
    def test(self):
        self.test2()
        #self.wait(event=Event(gain, Belief("A")))

    @pl(gain, Goal("ask"))
    def ask_stuff(self, src):
        self.print("asking for stuff")
        self.test()
        answer = self.send('ag2', askOneReply, Belief("stuff","Num"))
        self.print("the reply was ",answer)
        Admin().stop_all_agents()
        
    @pl(gain, Goal("meh"))
    def meh(self, src):
        self.add(Goal("koisas"),False)
        self.print("WATS")
        #self.print_intentions
        #self.print_goals
        self.wait(2)
        self.print("Heys")
        self.send('ag1',tell,Belief("A"))
        
    @pl(gain, Goal("koisas"))
    def koisas(self, src):
        self.print("works")
        #self.print_intentions
        #self.print("why")
    
    @pl(gain, Goal("action"))
    def action(self, src):
        self.print("hello")
        self.func_env()
        self.stop_cycle()
        

def hue(var: list[tuple[int,int]]):
    x, y = var.pop()
    return x, y


        


# TESTAR DESLIGAR E LIGAR CICLO DE AGENTES
# TESTAR LIGAR NOVAMENTE O SISTEMA APÒS O OUTRO TERMINAR
# ARRUMAR CONTEXTO PARA SER UMA FUNÇÂO QUE RETORNA BOOLEANO

def check(condition: dict):
    for x, y  in condition.items():
        a = check(y[0]) if isinstance(y[0], dict) else y[0]
        b = check(y[1]) if isinstance(y[1], dict) else y[1]
        
        return f'{a} {x} {b}'

if __name__ == "__main__":
    from itertools import product
    actions = ([1,2,3],  ("up","down","left","right"))
    print(list(product(*actions)), list(range(5)))
    #a = Belief("C", Any) | ( Belief("A", Any) < Belief("B", Any) )
    
    #print(check(a))
    
    # env = Environment("Sample")
    # env.create(Percept("ab", [-1,1], interval))
    # perc = env.get(Percept("ab","Ab"))
    # assert isinstance(perc, Percept)
    # print(perc.group)
    # print(perc.group in Group._member_names_)
    # a = (3, 5) - (2, 1)
    # print(a)
    # exit()
    # #Admin().set_logging(show_exec=True, show_cycle=True, show_prct=True, show_slct=True)
    # ag1 = Ag("ag1")
    # ag1.add(Goal("action"))
    # env = Env("env")
    # Admin().connect_to(ag1, env)
    #ag2 = Ag("ag2")
    #ag1.add(Goal("ask"))
    #ag2.add(Belief("stuff",42))
    #ag2.add(Goal("meh"))
    #Admin().slow_cycle_by(0.000001)
    #Admin().start_system()