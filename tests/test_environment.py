from maspy.agent import Agent
from maspy.admin import Admin
from maspy.environment import Environment, Percept, DEFAULT_GROUP
from typing import List
import pytest

def test_clean():
    env = Environment("clean")
    assert type(env._clean(None)) is dict
    with pytest.raises(TypeError):
        env._clean(42)
    with pytest.raises(AttributeError):
        env._clean([42])
        
    env._clean([Percept("KEY",(1,)),
                Percept("KEY",(2,)),
                Percept("KEY",(3,)),
                Percept("NEW_KEY")])

def test_percepts_are_Percept():
    env = Environment("test_env1")
    with pytest.raises(Exception):
        env.create()
        
    env.create("KEY")
    env.create("KEY",5)
    env.create("KEY",(1,2,3),"Normal")
    env.create("KEY",group="Normal")
    env.create(percept=Percept("B"))
    env.create(percept=[Percept("C",group="Normal"), Percept("D",(2,))])
    for group, keys in env._percepts.items():
            for key, percept_set in keys.items():
                for percept in percept_set:
                    assert (type(percept) is Percept and percept.key == key and percept.group == group)
                    
def test_delete_percepts():
    env = Environment("test_env2")
    env.create("KEY",23,"Normal")
    env.create("KEY",(42,27))
    assert Percept("KEY",(23,),"Normal") in env._percepts["Normal"]["KEY"]
    assert Percept("KEY",(42,27)) in env._percepts[DEFAULT_GROUP]["KEY"]
    env.delete("KEY",23,"Normal")
    env.delete("KEY",(42,27))
    assert Percept("KEY",(23,),"Normal") not in env._percepts["Normal"]["KEY"]
    assert Percept("KEY",(42,27)) not in env._percepts[DEFAULT_GROUP]["KEY"]
    
def test_change_percept():
    env = Environment("change")
    env.create("KEY",42)
    env.change("KEY",42,27)
    assert Percept("KEY",(42,)) not in env._percepts[DEFAULT_GROUP]["KEY"]
    assert Percept("KEY",(27,)) not in env._percepts[DEFAULT_GROUP]["KEY"]

def test_percept_exists():
    env = Environment("test_env3")
    env.create("KEY",23)
    assert Percept("KEY",(23,)) in env._percepts[DEFAULT_GROUP]["KEY"] and \
        env._percept_exists("KEY",23)
    env.print_percepts

def test_perception():
    env = Environment("test_env1")
    perception_dict, perception_list = env.perception()
    assert perception_dict == env._percepts 
    for percept in perception_list:
        assert (type(percept) is Percept)
        
def test_environment_instances():
    env = Environment()
    env1 = Environment("env1")
    env2 = Environment("env2")
    assert env1 != env2 and env1 == Environment("env1") and env2 == Environment("env2") and env1 != env and env2 != env
    
def test_add_agents():
    ag1 = Agent("Simple")
    ag2 = Agent("Simple")
    ag_list = [Agent(f"Simple{i}") for i in range(10)]
    env = Environment("Adding")
    ag1.connect_to(env)
    ag2.connect_to(env)
    env.add_agents(ag_list)
    ag_list += [ag1,ag2]
    for agent in ag_list:
        assert agent.my_name in env.agent_list["Agent"][agent.my_name[0]]
        assert agent == env._agents[agent.my_name]