from maspy.agent import *
from maspy.communication import Channel
from maspy.admin import Admin
import pytest

def test_plan():
    pass

def test_add_rm_agent():
    ag1 = Agent("ag1")
    ag2 = Agent("ag2")
    Channel("Special").add_agents([ag1,ag2])
    ag1.send(ag2.my_name,tell,("KEY",),"Special")
    Channel("Special")._rm_agents([ag1,ag2])
    Channel()._rm_agents(ag2)
    ag1.send(ag2.my_name,tell,("KEY",27))

def test_send_msg():
    ag1 = Agent("ag1",instant_mail=True)
    ag2 = Agent("ag2",instant_mail=True)
    ag1.send(ag2.my_name,"sending",("KEY",))
    assert ag2.has_close(Belief("KEY")) == False
    ag1.send(ag2.my_name,tell,("KEY",))
    assert ag2.has_close(Belief("KEY")) == True
    ag1.send(ag2.my_name,tellHow,test_plan)
    ag1.send(ag2.my_name,askOne,("ASK",))
    ag1.send(ag2.my_name,untell,("KEY",))
    ag1.send(ag2.my_name,achieve,("OBJ",))
    ag1.send(ag2.my_name,unachieve,("OBJ",))
    ag1.send(ag2.my_name,tell,("KEY",),"VIP_Channel")

    
 