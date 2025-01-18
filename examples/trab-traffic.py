from maspy import *
from random import choice 

"""
Controlador: Controla o cruzamento para a travessia dos carros
VA: Carros querem atingir um destino e precisam cruzar o cruzamento
Ambiente(Cruzamento): Contem informações da posição dos carros nas ruas e suas intenções de destino

carros_na_rua: Armazena as fila de carros em uma determinada rua(A,B,C ou D)
carro_na_rua: Armazena o carro em uma rua que tem sua rota
carro_cruzamento: Armazena os carros atualmente no cruzamento

RUAS: Constante com as ruas disponíveis.
PRIORIDADES: Prioridades que carros podem assumir: Comum para carros de passeio, Urgencia para carros a caminho da trabalho,
        Transporte, para o transporte coletivo de passageiros, Emergencia para veículos de socorro.    i
"""


RUAS = [
    "rua_A",
"rua_B","rua_C",
    "rua_D",
]

PRIORIDADES = ["comum", "urgencia", "transporte", "emergencia"]

class Cruzamento(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.create(Percept("carros_cruzamento", ({})))
        
        for RUA in RUAS: 
            self.create(Percept("carros_na_rua", (RUA,[])))
        
    def entrar_na_rua(self, agt, nome_rua, rua_destino):
        fila_rua = self.get(Percept("carros_na_rua", (nome_rua, "Lista" ))) 
        carros_cruzamento = self.get(Percept("carros_cruzamento", "Dicionario" ))
        
        if fila_rua and carros_cruzamento:
            
            fila_rua.args[1].append(agt)
            self.change(fila_rua, fila_rua.args)
            self.print(f"{agt} entrou na rua {fila_rua.args[0]}: {fila_rua.args[1]}")

            if nome_rua not in carros_cruzamento.args:
                carros_cruzamento.args[nome_rua] = agt
                self.change(carros_cruzamento, carros_cruzamento.args)
                self.print(f"O {agt} entrou no cruzamento da rua {nome_rua}: {carros_cruzamento.args}")

        else:
            self.print(f"Erro: Estrada {nome_rua} não existe ou percepts estão inconsistentes!")
    
    def obter_status(self): # obtem o estado atual das ruas e cruzamento
            estado_ruas = {rua.value[0]: rua.value[1] for rua in self.get_all("carros_na_rua")}
            carros_cruzamento = self.get(Percept("carros_cruzamento", None)).value
            return {
                "ruas": estado_ruas,
                "carros_cruzamento": carros_cruzamento
            }
    
class VA(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
        self.add(Goal("Seguir_destino"))
        #self.connect_to(Environment("I1"))
        self._criar_crencas()
    
    @pl(gain, Goal("Seguir_destino"))
    def informar_rota(self, src):
        rua_atual = self.get(Belief("Rua", "Nome"))
        rua_destino = self.get(Belief("Direcao", "Rua"))

        if rua_atual and rua_destino:
            self.entrar_na_rua(rua_atual.args, rua_destino.args)
    
    def _criar_crencas(self):
        rua_escolhida = choice(RUAS)
        direcao = choice([RUA for RUA in RUAS if RUA is not rua_escolhida])
        prioridade = choice(PRIORIDADES)
        
        self.add(Belief("Rua", (rua_escolhida)))
        self.add(Belief("Direcao", (direcao)))
        self.add(Belief("Prioridade", (prioridade)))

        self.print(f"Crenças criadas, Rua: {rua_escolhida}, Direcao: {direcao}, Prioridade: {prioridade}")
        
    #@pl(gain, Belief("at_intersection"))
    #def arrive_destination(self, src):
    #    chosen_road = choice(["A","B","C","D"]) # decide aleatoriamente entre A ou B
    #    self.add(Belief("Road", chosen_road)) # atualiza a belief do agente com a estrada escolhida
    #    self.add(Goal("cross_intersection"))
    #    self.action("I1").chose_road(self.my_name, chosen_road) # chama o ambiente para registrar a escolha
    #    colocar atributos nesse plano? organizar melhor caso sim

 #   @pl(gain, Goal("cross_intersection"), Belief("at_intersection"))
#   def notify_controller(self, src):
#        self.add()
    
class Controlador(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)  
        self.add(Goal("keep_crossing"))
        self.add(Belief("order", [])) 
        self.add(Belief("cars_at_intersection", []))

if __name__=="__main__":
    i1 = Cruzamento("I1")
    c1 = Controlador("C1") # conectar no ambiente
    agents = [VA("VA") for i in range(1, 6)]

    Admin().connect_to(agents, i1)
    Admin().start_system()  