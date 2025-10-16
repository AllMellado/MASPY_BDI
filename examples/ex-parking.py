from maspy import *
import random as rnd
from time import sleep
from threading import Lock
import sys, json, math
import yappi

global_lock = Lock()
end_flag = 0
success_flag = 0

class Parking(Environment):
    def __init__(self, name, num=10):
        super().__init__(name)
        
        self.create(Percept("sold_spots",0,"Spots",False))
        pass # self.print(f"Starting parking with {nr_spots} spots")
        for n in range(1,num+1):
            self.create(Percept("spot",(n,"free"),"Spots",False))
    
    def park_spot(self, agt, spot_id):
        spot = self.get(Percept("spot",(spot_id,"free")))
        pass # self.print(f"Driver {agt} parking on spot({spot_id})")
        if spot:
            self.change(spot,(spot_id,[agt]))
            return True
        else:
            pass # self.print(f"Requested spot({spot_id}) unavailable")
            return None
    
    def leave_spot(self, agt):
        spot = self.get(Percept("spot",(Any,[agt])))
        if spot:
            pass # self.print(f"Driver {agt} leaving spot({spot.args[0]})")
            self.change(spot,(spot.args[0],"free"))
        else:
            pass # self.print(f"Driver {agt} not found in any spot")

class Manager(Agent):
    def __init__(self, agt_name=None):
        super().__init__(agt_name,read_all_mail=True)
        self.add(Belief("spotPrice",15,adds_event=False))
        self.add(Belief("minPrice",10,adds_event=False))
        self.add(Goal("broadcast_price"))
        pass # self.print("Manager initialized")
        self.end_counter = 0
    
    @pl(gain,Goal("broadcast_price"), Belief("spotPrice",Any))
    def send_price(self,src,spot_price):
        pass # self.print(f"Broadcasting spot price[{spot_price}] to all Agents in Parking Channel.")
        self.send(broadcast,Goal("checkPrice",spot_price),"Parking")
    
    @pl(gain,Goal("offer_answer",(Any,Any)), Belief("minPrice",Any))
    def offer_response(self,src,offer_answer,min_price):
        answer, price = offer_answer
        pass # self.print(f"Received answer[{answer}] from {src}")
        match answer:
            case "reject":
                pass # self.print(f"Given price[{price}] rejected by {src}")
                pass
            case "accept":
                pass # self.print(f"Price accepted[{price}] by {src}. Choosing spot.")
                self.add(Goal("SendSpot",src),True)
            case "offer":
                if price < min_price:
                    counter_offer = (min_price+price)/1.8
                    counter_offer = round(counter_offer,2)
                    pass # self.print(f"Price offered[{price}] from {src} too low. Counter-offer[{counter_offer}]")
                    self.send(src,Goal("checkPrice",counter_offer),"Parking")
                else:
                    pass # self.print(f"Offered price from {src} accepted[{price}]. Choosing spot.")
                    self.add(Goal("SendSpot",src),True)
    
    @pl(gain,Goal("SendSpot",Any), Belief("spot",(Any,"free"),"Parking"))        
    def send_spot(self, src, agent, spot):
        spot_id = spot[0]
        pass # self.print(f"{self.cycle_counter} Sending spot({spot_id}) to {agent}")
        self.send(agent,Goal("park",("Parking",spot_id)),"Parking")
    
    @pl(gain,Goal("SendSpot",Any))
    def unavailable_spot(self, src, agent):
        pass # self.print(f"{self.cycle_counter} No spots available for {agent}")
        self.send(agent,Belief("no_spots_available"),"Parking")
    
    
class Driver(Agent):
    def __init__(self, agt_name, budget, counter, wait):
        super().__init__(agt_name)
        self.counter = counter
        self.wait_time = wait
        self.last_price = 0
        self.add(Belief("budget",budget,adds_event=False))
    
    @pl(gain,Goal("checkPrice",Any),Belief("budget",(Any,Any)))
    def check_price(self,src,given_price,budget):
        self.wait(rnd.random()*2)
        want_price, max_price  = budget
        self.add(Belief("offer_made",given_price,adds_event=False))
        if self.last_price == given_price:
            pass # self.print(f"Rejecting price[{given_price}]. Same as last offer")
            answer = ("reject",given_price)
        elif given_price > 2*max_price:
            pass # self.print(f"Rejecting price[{given_price}]. Too Higher than my max[{max_price}]")
            answer = ("reject",given_price)
        elif given_price <= want_price:
            pass # self.print(f"Accepting price [{given_price}]. Wanted[{want_price}]")
            answer = ("accept",given_price)
        else:
            counter_offer = (want_price+given_price)/(self.counter+1.5)
            counter_offer = round(counter_offer,2)
            pass # self.print(f"Making counter-offer for price[{given_price}]. Offering[{counter_offer}]")
            answer = ("offer",counter_offer)
            
        if answer[0] == "reject": 
            self.end_reasoning()
        else:
            self.add(Belief("offer_answer",answer,adds_event=False))
            self.send(src,Goal("offer_answer",answer),"Parking")
            self.last_price = given_price
    
    @pl(gain,Belief("no_spots_available"))
    def no_spots(self,src):
        pass # self.print("Leaving because no spots!")
        self.end_reasoning()
        
    
    @pl(gain,Goal("park",(Any,Any)))
    def park_on_spot(self,src,spot):
        park_name, spot_id = spot
        self.connect_to(park_name)
        pass # self.print(f"Parking on spot({spot_id})")
        confirm = self.park_spot(spot_id)
        if confirm is None:
            pass # self.print(f"Spot is unavailable after given by {src}")
            self.end_reasoning()
            return None
        sleep(self.wait_time)
        pass # self.print(f"Leaving spot({spot_id})")
        self.leave_spot()
        self.disconnect_from(park_name)
        global success_flag 
        with self.lock:
            success_flag += 1
        self.end_reasoning()
        
    def end_reasoning(self):
        self.stop_cycle()
        global end_flag
        global global_lock
        with global_lock:
            if Admin().sys_running:
                end_flag += 1
                if end_flag % 10 == 0:
                    pass # print(f"{self.my_name} Ended reasoning ({end_flag})")
                if Admin().running_class_agents("Drv") is False:
                    Admin().stop_all_agents()

if __name__ == "__main__":
    yappi.start()
    success_flag = 0
    park = Parking('Parking',100)
    park_ch = Channel("Parking")
    manager = Manager()

    drv_settings: dict = {"budget": [(10,12),(8,14),(10,20),(12,14),(12,16)],
                    "counter": [0.4, 0.8, 1, 1.2, 1.4],
                    "wait": [0, 0.5, 0.7, 1, 1.5]}
    driver_list: list = []
    for i in range(100):
        budget = drv_settings["budget"][i%5]
        counter = drv_settings["counter"][(i*2)%5]
        wait = drv_settings["wait"][(i*4)%5]
        drv = Driver("Drv",budget,counter,wait)
        driver_list.append(drv)
        
    manager.read_all_mail = False
    Admin().connect_to(manager, [park,park_ch])
    Admin().connect_to(driver_list, park_ch)
    Admin().start_system()
    with open(r"T:\Downloads\measuring\duration.json","w") as f:
        json.dump({"elapsed_time": Admin().elapsed_time}, f)
    yappi.stop()
    yappi.get_func_stats().save("yappi.prof", type="pstat")