import inspect
import random
from functools import wraps
from agent import agent, Belief, Ask, Objective

class comms:
    def __init__(self, env_name='Comm') -> None:
        self.env_name = env_name
        self.__started_agents = []
        self.__agent_list = {}
        self.__agents = {}
        agent.send_msg = self.function_call(agent.send_msg)

    def add_agents(self, agents):
        try:
            for agent in agents:
                self.__add_agent(agent)
        except(TypeError):
            self.__add_agent(agents)

    def __add_agent(self, agent):
        agent.my_name = f'{agent.my_name}#{random.randint(1000,9999)}' 
        if agent.my_name in self.__agents:
            aux = agent.my_name.split('#')
            while "#".join(aux) in self.__agents:
                aux[-1] = str(random.randint(1000,9999))
                
            agent.my_name = '#'.join(aux) 
        self.__agent_list[agent.my_name] = type(agent).__name__
        self.__agents[agent.my_name] = agent
        print(f'{self.env_name}> Connecting agent {type(agent).__name__}:{agent.my_name} to environment')

    def rm_agents(self, agents):
        try:
            for agent in agents:
                self.__rm_agent(agent)
        except(TypeError):
            self.__rm_agent(agents)
        self.send_agents_list()
    
    def __rm_agent(self, agent):
        if agent.my_name in self.__agents:
            del(self.__agents[agent.my_name])
            del(self.__agent_list[agent.my_name])
        print(f'{self.env_name}> Desconnecting agent {type(agent).__name__}:{agent.my_name} from environment')
    
    def start_all_agents(self):
        no_agents = True

        self.send_agents_list()
        print(f'{self.env_name}> Starting all connected agents')
        for agent_name in self.__agents:
            no_agents = False
            self.__start_agent(agent_name)

        if no_agents:
            print(f'{self.env_name}> No agents are connected')

    def start_agents(self, agents):
        self.send_agents_list()
        try:
            print(f'{self.env_name}> Starting listed agents')
            for agent in agents:
                self.__start_agent(agent.my_name)
        except(TypeError):
            print(f'{self.env_name}> Starting agent {type(agents).__name__}:{agents.my_name}')
            self.__start_agent(agents.my_name)

    def __start_agent(self,agent_name):
        try:
            if agent_name in self.__started_agents:
                print(f"{self.env_name}> Agent {agent_name} already started")
                return
            self.__started_agents.append(agent_name)
            agent = self.__agents[agent_name]
            agent.reasoning()
        except(KeyError):
            print(f"{self.env_name}> Agent {agent_name} not connected to environment")

    def send_agents_list(self):
        for agent_name in self.__agents:
            self.__agents[agent_name].recieve_msg(agent_name,'env_tell',Belief('Agents',[self.__agent_list]))
            
    def function_call(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_values = inspect.getcallargs(func, *args, **kwargs)
            msg = {}
            for key,value in arg_values.items():
                msg[key] = value 
            try:
                self.__agents[msg['target']].recieve_msg(msg['self']\
                                .my_name,msg['act'],msg['msg'])
            except(KeyError):
                print(f"Agent {msg['target']} doesn't existsssss")

        
            return func(*args, **kwargs)
        
        return wrapper

