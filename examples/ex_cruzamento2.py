from maspy import *
import random


class Cruzamento(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.create(Percept("prioridade_atual", "None"))  # Inicializa a percepção

    def atualizar_prioridade(self, src, novo_veiculo):
        self.print(f"Atualizando prioridade para: {novo_veiculo}")
        prioridade = self.get(Percept("prioridade_atual", Any))
        if prioridade:
            self.change(prioridade, novo_veiculo)
        else:
            self.print("erro ao encontrar a percepção")
        #self.change(Percept("prioridade_atual", novo_veiculo))  # Atualiza a percepção no ambiente

    def liberar_passagem(self, src):
        self.print(f"Liberando passagem para: {src}")
        prioridade = self.get(Percept("prioridade_atual", Any))
        if prioridade:
            self.change(prioridade, "None")
        else:
            self.print("erro ao encontrar a percepção")
        #self.delete(Percept("prioridade_atual", src))  # Remove a percepção após a passagem


class VA(Agent):
    def __init__(self, agt_name, nivel_prioridade):
        super().__init__(agt_name)
        self.add(Belief("no_cruzamento"))
        self.add(Belief("nivel_prioridade", nivel_prioridade))
        self.add(Goal("atravessar_cruzamento"))


    @pl(gain, Goal("atravessar_cruzamento"), Belief("no_cruzamento"))
    def avaliar_prioridade(self, src):
        prioridade = self.get(Belief("prioridade_atual", Any, "Cruzamento"))
        if prioridade and prioridade.args == self.my_name:
            self.add(Goal("tem_prioridade"))
        else:
            self.add(Goal("sem_prioridade"))

    @pl(gain, Goal("sem_prioridade"))
    def esperar(self, src):
        self.print(f"{self.my_name} aguardando prioridade.")
        teste = self.get(Belief("nivel_prioridade", Any))
        dados = []
        dados.append(self.my_name)
        dados.append(teste.args)
        self.send("Controlador", tell, Belief("solicitar_prioridade", dados))

    @pl(gain, Goal("tem_prioridade"))
    def atravessar(self, src):
        self.print(f"{self.my_name} atravessando o cruzamento.")
        self.liberar_passagem()
        # Envia sinal que finalizou ao Controlador (opcional, se for controlar término)
        self.send("Controlador", tell, Belief("veiculo_finalizado", self.my_name))
        self.stop_cycle()
        

class Controlador(Agent):
    def __init__(self, agt_name):
        super().__init__(agt_name)
        self.queue = []  # Fila para gerenciar pedidos de prioridade
        # Controlar quais veículos ainda estão ativos (opcional)
        self.veiculos_ativos = set()

    @pl(gain, Belief("solicitar_prioridade", Any))
    def avaliar_pedido(self, src, dados):
        veiculo, nivel = dados
        self.print(f"Avaliando pedido de prioridade para: {veiculo} com prioridade: {nivel}")
        self.queue.append((veiculo, nivel))  # Adiciona o veículo na fila
        self.queue.sort(key=lambda x: x[1], reverse=True)
        self.processar_prioridade()

    def processar_prioridade(self):
        prioridade = self.get(Belief("prioridade_atual", Any, "Cruzamento"))
        self.print(f"fila de prioridade {self.queue}")
        if prioridade and prioridade.args != "None":
            return
        if self.queue:
            veiculo_atual, nivel = self.queue.pop(0)  # Remove o veículo da fila
            self.print(f"Prioridade concedida para: {veiculo_atual} com prioridade: {nivel}")
            self.atualizar_prioridade(veiculo_atual)
            self.send(veiculo_atual, achieve, Goal("tem_prioridade"))  # Notifica o veículo

    @pl(gain, Belief("veiculo_finalizado", Any))
    def veiculo_finalizado(self, src, veiculo):
        """
        Quando um veículo finaliza, remove-o do conjunto de ativos
        e tenta processar o próximo da fila (se existir).
        """
        self.print(f" {veiculo} finalizou.")
        if veiculo in self.veiculos_ativos:
            self.veiculos_ativos.remove(veiculo)

        # Processa o próximo de maior prioridade na fila (se houver)
        self.processar_prioridade()

        # Se não há mais veículos ativos e não há mais na fila, podemos encerrar
        if not self.veiculos_ativos and not self.queue:
            self.print("[CONTROLADOR] Todos veículos finalizados. Encerrando...")
            self.stop_cycle()
            #Admin().stop_all_agents()
            


if __name__ == "__main__":
    cruzamento = Cruzamento("Cruzamento")
    # 2) Define um número aleatório de veículos
    numero_veiculos = random.randint(2, 5)  # Ex: entre 2 e 5
    veiculos = []

    controlador = Controlador("Controlador")

    veiculos.append(controlador)
    for i in range(3):
        nome = f"VA{i+1}"
        prioridade = random.randint(1,3)
        veiculos.append(VA(nome, prioridade))
    # veiculo1 = VA("VA1")
    # veiculo2 = VA("VA2")
    # veiculo3 = VA("VA3")
    # veiculo4 = VA("VA4")
    # veiculo5 = VA("VA5")
    # veiculo6 = VA("VA6")

    comunicacao_ch = Channel("V2C")
    Admin().connect_to(veiculos, [comunicacao_ch, cruzamento])
    Admin().start_system()