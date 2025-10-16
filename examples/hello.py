from maspy import *

class Hello(Agent):
    @pl(gain,Goal("hello"))
    def say_hello(self,src):
        self.print("Hello World")
        self.stop_cycle()

Hello("Hello_Agent", goals=Goal("hello"))

Admin().start_system()