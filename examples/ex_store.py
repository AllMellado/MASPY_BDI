from maspy import *
from random import randint

class Store(Environment):
    def __init__(self, env_name = None):
        super().__init__(env_name)
        
    def add_product(self, src, name, qnt, value):
        product = self.get(Percept("product",("name","Q","V")))
        if product is None:
            self.create(Percept("product",(name,qnt,value)))
            self.print(f"{src} is creating product[{name}]({qnt}) selling for {value}")
        else:
            new_qnt = product.args[1]+qnt
            self.change(product,(name,new_qnt,value))
            self.print(f"{src} is adding ({qnt}) product[{name}]({new_qnt}) now selling for {value}")
    
    def sell_product(self, src, name):
        product = self.get(Percept("product",("name","Q","V")))
        if product is None:
            self.print(f"{src} tried selling unavailable product[{name}]")
        else:
            new_qnt = product.args[1]-1
            value = product.args[2]
            if new_qnt == 0:
                self.delete(product)
            else:
                self.change(product,(name,new_qnt,value))
            self.print(f"{src} selling product[{name}] for {value} - new qnt({new_qnt})")
    
class Seller(Agent):
    def __init__(self, agt_name, products):
        super().__init__(agt_name)
        self.products = products
        self.add(Goal("Restock"))
    
    @pl(gain, Goal("Restock"))
    def restock_store(self, src):
        pass
        
class Buyer(Agent):
    def __init__(self, agt_name, budget):
        super().__init__(agt_name)
        
        

if __name__ == "__name__":
    products = ["Fruits", "Vegetables", "Meats", "Seafoods", "Dairy", "Baked Goods", "Canned Goods", "Grains", "Cereals", "Snack", "Beverage", "Frozen Foods", "Condiment", "Sauce", "Spice", "Seasoning", "Cleaning"]
    

    sellers = [Seller("Seller",products) for _ in range(2)]
    buyers = [Buyer("Buyer",randint(20,50)) for _ in range(5)]
    store = Store()
    
    Admin().connect_to(sellers, store)
    #Admin().system_report = True
    Admin().start_system()
    

