from maspy import *
import random


class AmbienteDeTransito(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.create(Percept("tem congestionamento"))

    def desviar_congestionamento(self, src):
        desvio = random.choice([True, False])
        if desvio:
            self.print("Congestionamento desviado com sucesso!")
            self.create(Percept("congestionamento resolvido"))
        else:
            self.print("Não foi possível desviar o congestionamento.")
            self.create(Percept("congestionamento não resolvido"))


class CarroAutonomo(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
        self.add(Belief("dirigindoCarro"))

    @pl(gain, Belief("dirigindoCarro"))
    def verificar_trafego(self, src):
        self.print("Verificando condições de tráfego...")
        percepcao1 = self.get(Belief("tem congestionamento", source="AvenidaPrincipal"))

        if percepcao1:
            self.print("Congestionamento encontrado!")
            self.tentar_redirecionamento()

        self.perceive("AvenidaPrincipal")

    def tentar_redirecionamento(self):
        rota_alternativa = random.choice([True, False])
        if rota_alternativa:
            self.print("Rota alternativa disponível!")
        else:
            self.print("Sem rota alternativa disponível!")
            self.add(Goal("solicitarAjusteSemaforo"))

    @pl(gain, Goal("solicitarAjusteSemaforo"))
    def avisar_ControladorSemaforo(self, src):
        self.print("Solicitando ajuste de semáforo...")
        self.send("ControladorSemaforo", tell, Belief("AjusteSemaforoNecessario"), "C2T")

    @pl(gain, Belief("SemaforoAjustado"))
    def semaforo_ajustado(self, src):
        self.print("Semáforo ajustado! Continuando com o trajeto...")


class PedestreAutonomo(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
        self.add(Belief("querAtravessar"))

    @pl(gain, Belief("querAtravessar"))
    def solicitar_travessia(self, src):
        self.print("Pedestre solicitando travessia segura...")
        self.send("ControladorSemaforo", tell, Belief("TravessiaPedestreNecessaria"), "C2T")

    @pl(gain, Belief("TravessiaPermitida"))
    def atravessar(self, src):
        self.print("Semáforo ajustado para pedestres! Atravessando com segurança...")


class ControladorSemaforo(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)

    @pl(gain, Belief("AjusteSemaforoNecessario"))
    def ajustar_ciclo_semaforo(self, src):
        self.print("Ajustando ciclo de semáforo para melhorar o fluxo de tráfego...")
        self.send(src, tell, Belief("SemaforoAjustado"), "C2T")

    @pl(gain, Belief("TravessiaPedestreNecessaria"))
    def permitir_travessia(self, src):
        self.print("Ajustando semáforo para permitir a travessia de pedestres...")
        self.send(src, tell, Belief("TravessiaPermitida"), "C2T")


if __name__ == "__main__":
    ambiente_transito = AmbienteDeTransito("AvenidaPrincipal")
    carros = [CarroAutonomo(f'CarroAutonomo') for _ in range(100)]
    pedestre1 = PedestreAutonomo("PedestreAutonomo1")
    controlador_semaforo = ControladorSemaforo("ControladorSemaforo")
    comunicacao_ch = Channel("C2T")
    Admin().connect_to([*carros, pedestre1, controlador_semaforo], [comunicacao_ch, ambiente_transito])
    Admin().start_system()
