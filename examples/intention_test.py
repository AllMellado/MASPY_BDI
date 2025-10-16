from maspy import Agent, pl, gain, Goal, Admin

class PrintAgent(Agent):
    @pl(gain,Goal("1"))
    def goal1(self,src):
        self.print("Goal 1 print 2.")
        self.print("Goal 1 print 3.")
        self.print("Goal 1 print 4.")
        self.print("Goal 1 print 5.")
        self.print("Goal 1 print 6.")
        
    @pl(gain,Goal("2"))
    def goal2(self,src):
        self.print("Goal 2 print 2.")
        self.print("Goal 2 print 3.")
        self.print("Goal 2 print 4.")
        self.print("Goal 2 print 5.")
        self.print("Goal 2 print 6.")
    
    @pl(gain,Goal("3"))
    def goal3(self,src):
        self.print("Goal 3 print 2.")
        self.print("Goal 3 print 3.")
        self.print("Goal 3 print 4.")
        self.print("Goal 3 print 5.")
        self.print("Goal 3 print 6.")
        
if __name__ == "__main__":
    ag = PrintAgent()
    ag.max_intentions = 3
    ag.add(Goal("1"))
    ag.add(Goal("2"))
    ag.add(Goal("3"))
    Admin().start_system()
    
    