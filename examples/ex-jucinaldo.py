from maspy import *
from maspy.learning import *
from time import sleep
import random
import threading
import time
from string import Template

# --- Controlador de Tráfego ---
class TrafficController(Agent):
    def __init__(self, agt_name, cars, model):
        super().__init__(agt_name)
        self.cars = cars
        self.model = model

    @pl(gain, Belief("start_traffic"))
    def manage_traffic(self, src):
        """Controlador inicia o tráfego e ordena que os carros comecem a se mover"""
        self.print(">>> Controlador iniciando gerenciamento de tráfego...")
        for car in self.cars:
            try:
                self.send(car.my_name, achieve, Goal("start_moving"))
            except Exception as e:
                self.print(f"Erro ao enviar start_moving para {car.my_name}: {e}")

# --- Carros Autônomos ---
class AutonomousCar(Agent):
    def __init__(self, agt_name, model, road_length, num_lanes):
        super().__init__(agt_name)
        self.add_policy(model)  # Adiciona o modelo de aprendizado ao carro
        self.auto_action = True
        self.road_length = road_length
        self.num_lanes = num_lanes
        self.position = (random.randint(0, road_length - 1), random.randint(1, num_lanes))  # Posição inicial aleatória

    @pl(gain, Goal("start_moving"))
    def move_car(self, src):
        """O carro escolhe uma ação com base na política aprendida e se move"""
        self.print(f">>> {self.my_name} iniciando movimento...")
        self.best_action("Traffic")
        # action = self.choose_action()
        # self.print(f"{self.my_name} escolheu a ação {action}")

        # try:
        # # Busca o ambiente "Traffic" corretamente
        #     traffic_env = next((env for env in Admin()._environments.values() if env.my_name == "Traffic"), None)

        #     if traffic_env:
        #         traffic_env.move(self.my_name, action)  # CHAMADA CORRIGIDA
        #     else:
        #         raise RuntimeError(f"Erro: {self.my_name} não encontrou o ambiente Traffic.")

        # except Exception as e:
        #     self.print(f"Erro ao mover {self.my_name}: {e}")



    def choose_action(self):
        """Seleciona uma ação aleatória baseada na política de aprendizado"""
        return random.choice(["up", "down", "left", "right"])

# --- Ambiente de Tráfego ---
class TrafficEnvironment(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.num_lanes = 3
        self.road_length = 15
        self.target = self.road_length
        print(">>> Ambiente de tráfego inicializado.")

        possible_positions = [(pos, lane) for pos in range(0, self.road_length + 1)
                              for lane in range(1, self.num_lanes + 1)]
        self.create(Percept("position", possible_positions, listed))
        self.create(Percept("reward", 0))
        
    def move_transition(self, state: dict, action: str):
        """Define a transição de estado do veículo no ambiente."""
        if "position" not in state:
            return state, -10, False

        pos, lane = state["position"]
        reward, terminated = -1, False

        if action == "up" and pos < self.road_length:
            pos += 1
        elif action == "down" and pos > 0:
            pos -= 1
        elif action == "left" and lane > 1:
            lane -= 1
        elif action == "right" and lane < self.num_lanes:
            lane += 1

        if pos >= self.target:
            reward += 50
            terminated = True

        return {"position": (pos, lane)}, reward, terminated

    @action(listed, ["up", "down", "left", "right"], move_transition)
    def aa(self, agt, direction: str):
        """Executa o movimento do carro."""
        try:
            self.print(f"{agt} está se movendo {direction}")
            sleep(1)  # Simula o tempo de movimentação
        except Exception as e:
            self.print(f"Erro ao mover {agt}: {e}")

# --- Configuração e Inicialização ---
def iniciar_sistema():
    #try:
    traffic_env = TrafficEnvironment("Traffic")
    #Admin()._add_environment(traffic_env)
    print(traffic_env.num_lanes)
    # Criando o modelo de aprendizado
    model = EnvModel(traffic_env)

    # Aplicando aprendizado por reforço (Q-Learning)
    print(">>> Iniciando aprendizado por reforço...")
    model.learn(qlearning, max_steps=100, num_episodes=1000)
    print(">>> Aprendizado finalizado.")


    cars = [AutonomousCar(f"Carro{i+1}", model, traffic_env.road_length, traffic_env.num_lanes) for i in range(10)]

    # Criando controlador de tráfego
    controller = TrafficController("Controlador", cars, model)

    # Conectar agentes ao ambiente
    Admin().connect_to([controller] + cars, traffic_env)

    # Iniciar o tráfego
    controller.add(Belief("start_traffic"))

    # Iniciar o sistema MASPY
    Admin().start_system()
    print(">>> Sistema iniciado com sucesso.")

    #except Exception as e:
    #    print(f"Erro fatal durante a inicialização do sistema: {e}")

# --- Execução ---
if __name__ == "__main__":
    iniciar_sistema()
