
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

class env:
    def __init__(self, env_name='env') -> None:
        self.__my_name = env_name
        self.__facts = {'any' : {}}
        self.__roles = {'any'}

    def add_role(self, role_name):
        if type(role_name) == str:
            self.__roles.add(role_name)
        else:
            print(f'{self.__my_name}> role *{role_name}* is not a string')

    def rm_role(self, role_name):
        self.__roles.remove(role_name)

    def get_roles(self):
        return self.__roles
    
    def check_role(self, role):
        return role in self.__roles
    
    #def add_multiple_facts(self, name, data, role='any'):
    
    def add_fact(self, name, data, role='any'):
        if role != 'any':
            if role not in self.__roles:
                self.add_role(role)
        try:
            match data:
                case list() | tuple():
                    self.__facts[role][name].append(data)
                case dict() | set():
                    self.__facts[role][name].update(data)
                case int() | float():
                    self.__facts[role][name] = data
        except(KeyError):
            try:
                self.__facts[role][name] = data
            except(KeyError):
                self.__facts[role] = {name : data}
    
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
