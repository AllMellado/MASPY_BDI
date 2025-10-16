from maspy import *
    
    Percept( "spot" , (12,"free") ,  "Spots"  )
    
    
     Belief( "spot" , (12,"free") , "Parking" )
    

    Belief(    "spot"    , (12,"free") , "Parking" )
    Belief(  "Receiver"  ,             ,  "self"   )
      Goal( "SendSpot"   , "Driver_5"  ,  "self"   )
      Goal("receive_info",   message   , "Sender"  )
    
    @pl(gain,Goal("SendSpot", Any), Belief("spot",(Any,"free"),"Parking"))   
    def send_spot(self, src, agent, spot):
        spot_id = spot[0]
        self.reserve_spot(spot_id)
        self.print(f"{self.cycle_counter} Sending spot({spot_id}) to {agent}")
        self.send(agent, achieve, Goal("park",("Parking",spot_id)), "Parking")
    
    @pl(gain, Goal("receive_info", Any), Belief("receiver"))
    def recv_info(self, src, msg):
        self.print(f"Information [{msg}] - Received from {src}")
        self.send(src, tell, Belief("info_received"))
        self.stop_cycle()
        
    self.send(agent, achieve, Goal("park",("Parking",spot_id)), "Parking")    
    self.send( src ,  tell  , Belief("info_received")  )
    
    
    def park_spot(self, agt, spot_id):
        spot = self.get(Percept("spot", (spot_id,"reserved")))
        if spot:
            print(f"Driver {agt} parking on spot({spot_id})")
            self.change(spot, (spot_id, agt))
            return True
        else:
            print(f"Requested spot({spot_id}) unavailable")
            return False

class Blackboard(Environment):
    def __init__(self, env_name = None):
        super().__init__(env_name)
    
    def set_starter_bid(self, agt, product, value):
        pass
    
    def get_best_bid(self, agt, product):
        pass

    def make_bid(self, agt, product, value):
        pass

class Seller(Agent):
    def __init__(self, agt_name = None):
        super().__init__(agt_name)

class Buyer(Agent):
    def __init__(self, agt_name = None):
        super().__init__(agt_name)
        
if __name__ == "__main__":
    nr_sellers = 4
    nr_buyers = 20
    
    sellers = [Seller() for _ in range(nr_sellers)]
    buyers = [Buyer() for _ in range(nr_buyers)]
    
    agent_list = sellers + buyers
    board = Blackboard()
    Admin().connect_to(agent_list, board)
    
    

from maspy import *

class Sample(Agent):
    @pl(gain, Goal("print"))
    def Sample_plan(self, src):
        self.print(f"Running the Agent {src} Plan")
        self.stop_cycle()
    
    @pl(gain, Goal("send_info", Any), Belief("sender"))
    def send_info(self, src, msg):
        agents_list = self.list_agents("Sample")
        for agent in agents_list:
            if agent == self.my_name:
                continue
            self.print(f"Sending> {msg} to {agent}")
            self.send(agent, achieve, Goal("receive_info", msg))
            
        agents_list = self.list_agents("Test")
        for agent in agents_list:
            plan = self.get(Plan, Goal("print"))
            self.send(agent, tellHow, plan)
            self.send(agent, achieve, Goal("print"))
            
        self.stop_cycle()

    @pl(gain, Goal("receive_info", Any), Belief("receiver"))
    def recv_info(self, src, msg):
        self.print(f"Information [{msg}] - Received from {src}")
        self.send(src, tell, Belief("info_received"))
        self.stop_cycle()

if __name__ == "__main__":
    Channel().show_exec = True
    receiver = Sample("Receiver", Belief("receiver"))
    sender = Sample("Sender", goals=Goal("send_info","Hello"))    
    sender.add(Belief("sender"))
    Admin().start_system()


    

        