#-----------------------------------------------------
# Agente-MASPY-06: VA, Autoridade Externa e Ambiente
# comunicacao entre agentes e interacao c/ ambiente
#-----------------------------------------------------

from maspy import *

import random

class MonitoramentoUrbano(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.create(Percept("tem_obstaculo"))
    
    def desviar_obstaculo(self, src):
        desvio = random.choice([True, False])
        print(desvio)
        if desvio == True:
            self.print("Obstaculo desviado!")
            self.create(Percept("obstaculo_resolvido"))
        else:            
            self.print("Obstaculo nao desviado!")
            self.create(Percept("obstaculo_nao_resolvido"))
        #self.print_percepts

class VA(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
        self.add(Belief("conduzindoVA"))

    @pl(gain,Belief("conduzindoVA"))
    def verifica_via_urbana(self,src):
        self.print("Verificar via urbana")
        percepcao1 = self.get(Belief("tem_obstaculo",source="BR101"))
        self.print(percepcao1.key)       
        if percepcao1:                   
            self.action("BR101").desviar_obstaculo(self.my_name)
            
        self.perceive("all")
        
        percepcao2 = self.get(Belief("obstaculo_resolvido",source="BR101"))
        if percepcao2:
            self.print("Desvio realizado com sucesso!")
            self.stop_cycle()
        else:
            percepcao3 = self.get(Belief("obstaculo_nao_resolvido",source="BR101"))
            if percepcao3:
                self.add(Belief("obstaculo"))                   
                self.add(Goal("manobraCritica"))

    @pl(gain,Goal("manobraCritica"),Belief("obstaculo"))
    def avisar_stakeholder(self,src):
        # envia msg para o agente Stakeholder
        self.print("Manobra critica; Acionar stakeholder")
        self.send("ControladorRemoto",tell,Belief("VAprecisaAjuda"),"V2C")                                

    @pl(gain,Goal("manobraEmergencia"))
    def executar_manobra(self,src):
        self.print(f"Manobra executada conforme stakeholder: {src}")
        self.stop_cycle()

class Stakeholder(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
    
    @pl(gain,Belief("VAprecisaAjuda"))
    def enviar_manobra(self,src):
        # envia msg para agente Waymo (VA)
        self.send("Waymo", achieve, Goal("manobraEmergencia"),"V2C")
        self.stop_cycle()

if __name__ == "__main__":    
    #Admin().set_logging(True,True,True,True,True)
    monitor = MonitoramentoUrbano("BR101")
    veiculo = VA("Waymo")
    controlador = Stakeholder("ControladorRemoto")
    comunicacao_ch = Channel("V2C")
    Admin().connect_to([veiculo,controlador],[comunicacao_ch,monitor])
    Admin().start_system()