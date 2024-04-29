from maspy import *

class Crossing(Environment):
    def __init__(self, env_name):
        super().__init__(env_name)
        self.create_percept("traffic_light","Green")
    
    def cross(self, src):
        self.print(f"Agent {src.my_name} is now crossing")

class Cross_Manager(Agent):
    def __init__(self, mg_name): super().__init__(mg_name,full_log=False)
        
    @pl(gain,Belief("traffic_light",("Color",)))
    def traffic_light(self,src,color):
        vehicles = self.find_in("Vehicle","Env","Cross_Junction")
        for vehicle in vehicles["Vehicle"]:
            self.print(f"Detected traffic light: {color} in env {src} - sending signal to {vehicle}")
            self.send(vehicle,achieve,Goal("crossing_over"),"Crossing")
    
    @pl(gain,Belief("leaving_junction"))
    def end_watch(self,src):
        self.stop_cycle()

class Vehicle(Agent):
    def __init__(self, vh_name): super().__init__(vh_name,full_log=False)
    
    @pl(gain,Goal("crossing_over"))
    def crossing(self,src):
        self.print(f"Confirmation for crossing by {src}")
        self.execute_in("Cross_Junction").cross(self)
        self.print(f"Crossing Completed")
        self.send(src,tell,Belief("leaving_junction"),"Crossing")
        self.stop_cycle()


if __name__ == "__main__":
    cross_channel = Channel("Crossing")
    cross_env = Crossing("Cross_Junction")
    cross_manager = Cross_Manager("Cross_Manager")
    vehicle = Vehicle("Vehicle")
    Admin().connect_to([cross_manager,vehicle],[cross_channel,cross_env])
    Admin().start_all_agents()