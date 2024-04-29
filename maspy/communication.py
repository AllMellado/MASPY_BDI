import inspect
import random
from threading import Lock
from typing import List, Optional, Union, Dict, Set, Tuple, Any


class CommsMultiton(type):
    _instances: Dict[str, "Channel"] = {}
    _lock: Lock = Lock()

    def __call__(cls, __my_name="default"):
        with cls._lock:
            if __my_name not in cls._instances:
                instance = super().__call__(__my_name)
                cls._instances[__my_name] = instance
        return cls._instances[__my_name]


class Channel(metaclass=CommsMultiton):
    def __init__(self, comm_name) -> None:
        from maspy.agent import Belief, Goal, Ask, Plan
        self.data_types = {Belief,Goal,Ask,Plan}
        self._my_name = comm_name
        self.agent_list = {}
        self._agents = {}
        self._name = f"{type(self).__name__}:{self._my_name}"
        self.full_log = True
        #Agent.send_msg = self.function_call(Agent.send_msg)
        
    def print(self,*args, **kwargs):
        return print(f"{self._name}>",*args,**kwargs)
    
    def add_agents(self, agents):
        try:
            for agent in agents:
                self._add_agent(agent)
            #self.send_agents_list()
        except TypeError:
            self._add_agent(agents)

    def _add_agent(self, agent):
        if type(agent).__name__ in self.agent_list:
            if agent.my_name[0] in self.agent_list[type(agent).__name__]:
                self.agent_list[type(agent).__name__][agent.my_name[0]].update({agent.my_name})
                self._agents[agent.my_name] = agent
            else:
                self.agent_list[type(agent).__name__].update({agent.my_name[0] : {agent.my_name}})
                self._agents[agent.my_name] = agent
        else:
            self.agent_list[type(agent).__name__] = {agent.my_name[0] : {agent.my_name}}
            self._agents[agent.my_name] = agent
        
        self.print(f'Connecting agent {type(agent).__name__}:{agent.my_name}')

            
    def _rm_agents(self, agents):
        try:
            for agent in agents:
                self._rm_agent(agent)
        except TypeError:
            self._rm_agent(agents)
        #self.send_agents_list()

    def _rm_agent(self, agent):
        if agent.my_name in self._agents:
            del self._agents[agent.my_name]
            del self.agent_list[type(agent).__name__][agent.my_name[0]]
        self.print(
            f"Desconnecting agent {type(agent).__name__}:{agent.my_name}"
        )

    def _send(self, sender, target, act, msg):  
        #self.print(f"parsing {sender}:{act}:{msg}")
        msg = self.parse_sent_msg(sender,act,msg)
        
        self.print(f"{sender} sending {act}:{msg} to {target}") if self.full_log else ...        
    
        try:         
            self._agents[target].save_msg(sender,act,msg)
        except KeyError:
            self.print(f"Agent {target} not connected")
    
    def as_data_type(self, act, data):
        from maspy.agent import Belief, Goal
        match act:
            case "tell" | "env_tell" | "untell":
                return Belief(*data)
            case "achieve" | "unachieve":
                return Goal(*data)
            case _:
                self.print(f"Unknown Act {act}")
                return None

    def parse_sent_msg(self,sender, act, msg):
        from maspy.agent import Belief, Ask, Plan
        if type(msg) not in self.data_types:
            msg = self.as_data_type(act,msg)
        if type(msg) is not Plan and msg is not None:
            msg = msg.update(source=sender)
        if act in {"askOne","askAll","askHow"}:
            
            msg = Ask(msg, source=sender)
        return msg