from threading import Lock

'''
gerenciar 
    caracteristicas do ambiente
    artefato do ambiente
    cargo de agentes neste ambiente
    comunicacao do ambiente
'''

'''
Get perception:
    verificar situacao do ambiente
        -olhar todas caracteristicas
        -considerar cargo do agente

'''
class envMeta(type):
    _instances = {}
    _lock: Lock = Lock()
    
    def __call__(cls, __my_name='env'):
        with cls._lock:
            if __my_name not in cls._instances:
                instance = super().__call__(__my_name)
                cls._instances[__my_name] = instance
        return cls._instances[__my_name]

class env(metaclass=envMeta):
    def __init__(self, env_name) -> None:
        self._my_name = env_name
        self.__facts = {'any' : {}}
        self.__roles = {'any'}

    def add_role(self, role_name):
        if type(role_name) == str:
            self.__roles.add(role_name)
        else:
            print(f'{self._my_name}> role *{role_name}* is not a string')

    def rm_role(self, role_name):
        self.__roles.remove(role_name)

    def get_roles(self):
        return self.__roles
    
    def check_role(self, role):
        return role in self.__roles
    
    #def add_multiple_facts(self, name, data, role='any'):
    
    def create_fact(self, name, data, role='any'):
        if role != 'any':
            if role not in self.__roles:
                self.add_role(role)

        if role in self.__facts:
            if name not in self.__facts[role]:
                self.__facts[role] = {name : data}
            else:
                print(f"{self._my_name}> Fact *{name}:{role}* already created")
        else:
            self.__facts[role] = {name : data}


    def update_fact(self, name, data, role='any'):
        if self.fact_exists(name, role):
            self.__facts[role] = {name : data}

    def extend_fact(self, name, data, role='any'):
        if not self.fact_exists(name, role):
            return
        try:
            match self.__facts[role][name]:
                case list() | tuple():
                    self.__facts[role][name].append(data)
                case dict() | set():
                    self.__facts[role][name].update(data)
                case int() | float() | str():
                    print(f"{self._my_name}> Impossible to extend fact {name}:{type(data)}")
        except(TypeError, ValueError):
            print(f"{self._my_name}> Fact {name}:{type(self.__facts[role][name])} can't extend {type(data)}")
    
    def reduce_fact(self, name, data, role='any'):
        if not self.fact_exists(name, role):
            return
        try:
            match self.__facts[role][name]:
                case list() | set():
                    self.__facts[role][name].remove(data)
                case dict():
                    self.__facts[role][name].pop(data)
                case tuple():
                    old_tuple = self.__facts[role][name]
                    self.__facts[role][name] = tuple(x for x in old_tuple if x != data )
                case int() | float() | str():
                    print(f"{self._my_name}> Impossible to reduce fact {name}:{type(data)}")
        except(KeyError, ValueError):
            print(f"{self._my_name}> Fact {name}:{type(self.__facts[role][name])} doesn't contain {data}")
    
    def fact_exists(self, name, role):
        try:
            self.__facts[role][name]
        except(KeyError):
            print(f"{self._my_name}> Fact {name}:{role} doesn't exist")
            return False
        return True

    
    def rm_fact(self, del_name, del_data=None, del_role=None):
        if del_role is None:
            for role in self.__roles:
                if del_name in self.__facts[role].keys():
                    if del_data is None:
                        del self.__facts[role][del_name]
                    elif del_data in self.__facts[role][del_name].keys():
                        del self.__facts[role][del_name][del_data]

        
    def get_facts(self, agent_role=None):
        print(self.__facts)
        found_facts = {}
        for role in self.__roles:
            if role == agent_role or role == 'any' or agent_role == 'all':
                for fact in self.__facts[role]:
                    found_facts[fact] = self.__facts[role][fact]
        return found_facts
