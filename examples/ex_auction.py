from maspy import *

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
        