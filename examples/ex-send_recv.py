from maspy import *
import math, sys, json

class Sender(Agent):
    @pl(gain, Goal("send_info", Any))
    def send_info(self, src, msg):
        self.send(f"Receiver_{self.my_name.split("_")[1]}", achieve, Goal("receive_info", msg)) 
        self.stop_cycle()
        
class Receiver(Agent):
    @pl(gain, Goal("receive_info", Any))
    def recv_info(self, src, msg):
        pass # self.print(f"Message received with content: {msg} from {src}")
        self.send(src, tell, Belief("info_received"))
        self.stop_cycle()

if __name__ == "__main__":
    num_agents = math.ceil(int(sys.argv[1])/2)
    with open(r"T:\Downloads\measuring\duration.json","w") as f:
        json.dump({"running": "send_recv", "num_agents": num_agents}, f)
    for _ in range(num_agents):
        sender = Sender(goals=Goal("send_info", "Hello"))    
        receiver = Receiver()
    Admin().start_system()
    with open(r"T:\Downloads\measuring\duration.json","w") as f:
        json.dump({"elapsed_time": Admin().elapsed_time}, f)


























