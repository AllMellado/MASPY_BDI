from maspy import *

class Sample(Agent):
    @pl(gain, Goal("ask",Any), 
        ~Belief("value",Any) & ( (Belief("Test", Any) != "a") | ( Belief("Test2",(Any,5)) > 10 ) )  )
    def asking(self,src,name,test,test2,test3):
        self.print(f"asking {name} for value - test={test} test2={test2} test3={test3}")    
        value = self.send(name, askOneReply, Belief("value", Any))
        self.print(f"Got {value} from {value.source}")
    
    @pl(gain,Belief("value", Any))
    def got_value(self,src,value):
        self.print(f"Got {value} from {src}")

if __name__ == '__main__':
    ag1 = Sample("asking")
    ag2 = Sample("informant")
    ag1.add(Goal("ask","informant"))
    ag1.add(Belief("Test","b"))
    ag1.add(Belief("Test2",(24,5)))
    ag2.add(Belief("value",42))
    #Admin().report = True
    Admin().start_system()
    