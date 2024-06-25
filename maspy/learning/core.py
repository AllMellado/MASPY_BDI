from typing import Any, Dict, Optional, Set, List, Generic, Optional, Sequence, Tuple, Type, TypeVar, Union, SupportsFloat
from threading import Lock
from contextlib import closing
from time import sleep
from io import StringIO
from os import system
import numpy as np
import numpy.typing as npt
import sys
import math
import time

from maspy.learning.space import Space
from maspy.utils import np_random

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WrapperObsType = TypeVar("WrapperObsType")
WrapperActType = TypeVar("WrapperActType")
RenderFrame = TypeVar("RenderFrame")

class Model(Generic[ObsType, ActType]):
    def __init__(self, map = None):
        if map is None: # default 4x4 map
            map = [[-2, 0, 0,-2],
                   [-1, 0, 0, 0],
                   [ 0,-2,-1,-1],
                   [ 0, 0, 0,-2]]
        self.map = map
        self.desc = np.asarray(map, dtype=int)
        self.ori_desc = np.asarray(map, dtype=int).copy()
        
        locs = np.where(self.desc == -2)
        self.locs = [ (x,y) for x,y in zip(locs[0],locs[1]) ] 
        self.comb = 2**len(self.locs)

        self.num_rows, self.num_cols = self.desc.shape    
        self.num_states = self.num_rows * self.num_cols
        self.initial_state_distrib = np.zeros(self.num_states)
        self.max_row = self.num_rows - 1
        self.max_col = self.num_cols - 1
        self.num_actions = 4 # down, up, right, left
        self.s: int = -1
        self.P = {
            state: {action: [] for action in range(self.num_actions)}
            for state in range(self.num_states)
        }
        self.populate_states()

    metadata: dict[str, Any] = {"render_modes": []}
    render_mode: str | None = None
    # spec: ModelSpec # Not Implemented
    
    action_space: Space[ActType]
    observation_space: Space[ObsType]
    
    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    def populate_states(self):
        arrive_reward = 10
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                state = self.encode(row, col)
                if self.desc[row,col] == 0:   # Estados iniciais devem estar fora de sujeira e obstaculos
                    self.initial_state_distrib[state] += 1
                for action in range(self.num_actions):
                    new_row, new_col, = row, col
                    reward = 0 # Recompensa para movimentar normalmente
                    terminated = False
                    if action == 0 and row < self.max_row and self.desc[row+1, col] != -1 : #  Movimenta p/ Baixo
                        new_row = min(row+1,self.max_row)
                        if self.desc[new_row,col] == -2:    # Recompensa quando alcança posicao alvo
                            reward = arrive_reward
                    elif action == 1 and row > 0 and self.desc[row-1, col] != -1: # Movimenta p/ Cima
                        new_row = max(row-1, 0)
                        if self.desc[new_row,col] == -2:    # Recompensa quando alcança posicao alvo
                            reward = arrive_reward
                    elif action == 2 and col < self.max_col and self.desc[row, col+1] != -1: # Movimenta p/ Direita
                        new_col = min(col+1,self.max_col)
                        if self.desc[row,new_col] == -2:    # Recompensa quando alcança posicao alvo
                            reward = arrive_reward
                    elif action == 3 and col > 0 and self.desc[row, col-1] != -1: # Movimenta p/ Esquerda
                        new_col = max(col-1,0) 
                        if self.desc[row,new_col] == -2:    # Recompensa quando alcança posicao alvo
                            reward = arrive_reward
                    
                    new_state = self.encode(new_row,new_col) # Novo estado apos acao
                    self.P[state][action].append( 
                            (1.0, new_state, reward, terminated) 
                        )
                    
        self.initial_state_distrib /= self.initial_state_distrib.sum()
        self.action_space = Discrete(self.num_actions)
        self.observation_space = Discrete(self.num_states)

    # Recuperar Estado Atual de Acordo com Observacao
    def encode(self, row, col):
        return  row * self.num_cols + col

    # Recuperar Observacao de Acordo com Estado Atual
    def decode(self, i):
        out = []
        out.append(i % self.num_rows)
        i = i // self.num_rows
        out.append(i)
        return reversed(out)

    # Mascara de Possiveis Acoes
    def action_mask(self, state: int):
        mask = np.zeros(self.num_actions, dtype=np.int8)
        row, col = self.decode(state)
        if row < self.max_row and self.desc[row+1, col] != -1: 
            mask[0] = 1
        if row > 0 and self.desc[row-1, col] != -1:
            mask[1] = 1
        if col < self.max_col and self.desc[row, col+1] != -1: 
            mask[2] = 1
        if col > 0 and self.desc[row, col-1] != -1:
            mask[3] = 1
        return mask

    # Retorna o resultado de uma determinada Acao 
    def look(self, a: ActType) -> Tuple[ObsType, float, bool, dict]:
        transitions = self.P[self.s][a]
        if len(transitions) > 1:
            i = self.categorical_sample([t[0] for t in transitions])
            p, s, r, t = transitions[i]
        else:
            p, s, r, t = transitions[0]

        return (int(s), r, t, {"prob": p, "action_mask": self.action_mask(s)})

    # Realiza uma Acao no Estado Atual
    def step(self, a: ActType) -> Tuple[ObsType, SupportsFloat, bool, dict[str, Any]]:
        raise NotImplementedError
        transitions = self.P[self.s][a]
    
        if len(transitions) > 1:
            i = self.categorical_sample([t[0] for t in transitions])
            p, s, r, t = transitions[i]
        else:
            p, s, r, t = transitions[0]

        row,col = self.decode(s)
        if self.desc[row][col] == -2:
            self.desc[row][col] = 0
        elif self.desc[row][col] == 0 and r > 0:
            r = 0
            
        self.s = s
        self.last_action = a
        return (int(s), r, t, {"prob": p, "action_mask": self.action_mask(s)})

    # Transforma o caminho de Estados em sequencia de Acoes
    def states_to_actions(self, path):
        path_actions = []
        x, y = self.decode(self.s)
        for state in path:
            row, col = self.decode(state)
            dif = np.subtract((x,y),(row,col))
            if dif[0] != 0:
                if dif[0] < 0:
                    path_actions.append(0) # Baixo
                else:
                    path_actions.append(1) # Cima
            elif dif[1] != 0:
                if dif[1] < 0:
                    path_actions.append(2) # Direita
                else:
                    path_actions.append(3) # Esquerda
            else:
                return None                # Error
            x = row
            y = col
        #print(f'{self.s} {path} -> {path_actions}')
        return path_actions

    def actions_to_states(self,actions):
        states = []
        for action,value in enumerate(actions):
            if value == 1:
                s,_,_,_ = self.look(action)
                states.append(s)
        return states
    
    # Retorna possiveis Estados vizinhos e as melhores Acoes
    def sensor(self):
        best_actions = np.zeros((self.num_actions),dtype=int)
        possible_states = []
        row,col = self.decode(self.s)
        mask = self.action_mask(self.s)
        for i in range(max(row-1,0),min(row+1,self.max_row)+1):
            for j in range(max(col-1,0),min(col+1,self.max_col)+1):
                if self.desc[i,j] >= 0 and np.abs((row+col)-(i+j)) == 1:
                    possible_states.append(self.encode(i,j))
                
                if self.desc[i,j] == -2:
                    if i > row and mask[0] == 1:
                        best_actions[0] += 1
                    elif i < row and mask[1] == 1:
                        best_actions[1] += 1
                    if j > col and mask[2] == 1:
                        best_actions[2] += 1
                    elif j < col and mask[3] == 1:
                        best_actions[3] += 1
        return possible_states, best_actions
                
    # Resetar Estado para um Possivel Incial
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> Tuple[ObsType, dict[str, Any]]: # type: ignore
        if seed is not None:
            self._np_random, self._np_random_seed = np_random(seed)
        """ 
        self.s = self.categorical_sample(self.initial_state_distrib)
        self.desc = self.ori_desc.copy()
        self.last_action = None
        
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}
 """
    # Coloca o sistema em um Estado especifico
    def set_state(self, state) -> Tuple[ObsType, dict]: 
        self.s = state
        self.last_action = None
        
        return int(self.s), {"prob": 1.0, "action_mask": self.action_mask(self.s)}

    # Imprimir na tela uma representacao do cenario no estado atual
    def render(self) -> RenderFrame | list[RenderFrame] | None: #adj_matrix=None):
        raise NotImplementedError
        desc = self.desc.copy().tolist()
        outfile = StringIO()

        row, col = self.decode(self.s)
    
        # Representacao do aspirador no cenario 
        desc[row][col] = 1

        end = out = '+'+ ''.join(['-' for x in range(2*len(desc)-1)]) + '+\n'
        for k,line in enumerate(desc):
            out += '|'
            for i,x in enumerate(line):
                if x == -2:
                    out += '#'
                if x == -1:
                    out += '■'
                if x == 1:
                    out += 'O'
                if x == 0:
                    state = self.encode(k,i)
                    if len(adj_matrix) > 0 and adj_matrix[state,state] > 0:
                        if adj_matrix[state,state] == 1:
                            out += '*'
                        elif adj_matrix[state,state] == 2:
                            out += '~'
                        elif adj_matrix[state,state] == 3:
                            out += 'X'
                        else:
                            out += 'ADJ_ERROR'
                    else:
                        out += ' '
                if i < len(line)-1:
                    out += ':'
            out += '|\n'
        out += end

        outfile.write(out)
        if self.last_action is not None:
            outfile.write(f"({['Baixo','Cima','Direita','Esquerda'][self.last_action]})\n")
        else:
            outfile.write("\n")

        with closing(outfile):
            return outfile.getvalue()
    
    def close(self):
        pass

    @property
    def unwrapped(self):
        """Return the base non-wrapped model instance.
        
        Returns:
            Model: The base non-wrapped :class:`maspy.learning.Model` instance.  
        """
        return self
    @property
    def np_random_seed(self) -> int:
        if self._np_random_seed is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random_seed
    
    @property
    def np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random, self._np_random_seed = np_random()
        return self._np_random
    
    @np_random.setter
    def np_random(self, value: np.random.Generator) -> None:
        self._np_random = value
        self._np_random_seed = -1

    def __str__(self):
        """Returns a string of the environment with :attr:`spec` id's if :attr:`spec.

        Returns:
            A string identifying the environment
        """
        if self.spec is None:
            return f"<{type(self).__name__} instance>"
        else:
            return f"<{type(self).__name__}<{self.spec.id}>>"

    def __enter__(self):
        """Support with-statement for the environment."""
        return self

    def __exit__(self, *args: Any):
        """Support with-statement for the environment and closes the environment."""
        self.close()
        # propagate exception
        return False

    def has_wrapper_attr(self, name: str) -> bool:
        """Checks if the attribute `name` exists in the environment."""
        return hasattr(self, name)

    def get_wrapper_attr(self, name: str) -> Any:
        """Gets the attribute `name` from the environment."""
        return getattr(self, name)

    def set_wrapper_attr(self, name: str, value: Any):
        """Sets the attribute `name` on the environment with `value`."""
        setattr(self, name, value)


    """ def categorical_sample(self, prob_n):
        np_rand = np.random.default_rng()
        prob_n = np.asarray(prob_n)
        csprob_n = np.cumsum(prob_n)
        return np.argmax(csprob_n > np_rand.random())   """

class Wrapper(Model[WrapperObsType, WrapperActType],
              Generic[WrapperObsType, WrapperActType, ObsType, ActType]
        ):
    def __init__(self, env_model: Model[ObsType, ActType]):
        self.env_model = env_model
        assert isinstance(env_model, Model), f"Expected Model type, actual type: {type(env_model)}"
        
        self._action_space: Space[WrapperActType] | None = None
        self._observation_space: Space[WrapperObsType] | None = None
        self._metadata: dict[str, Any] | None = None
        
        # self._cached_spec: EnvSpec | None = None # Not Implemented
        
    def step(self, action: WrapperActType) -> Tuple[WrapperObsType, SupportsFloat, bool, dict[str, Any]]:
        return self.env_model.step(action)
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[WrapperObsType, dict[str, Any]]:
        return self.env_model.reset(seed=seed, options=options)
    
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env_model.render()
    
    def close(self):
        return self.env_model.close()
    
    @property
    def np_random_seed(self) -> int | None:
        return self.env_model.np_random_seed
    
    @property
    def unwrapped(self) -> Model[ObsType, ActType]:
        """Return the base non-wrapped model instance.
        
        Returns:
            Model: The base non-wrapped :class:`maspy.learning.Model` instance.  
        """
        return self.env_model.unwrapped
    
    @property
    def spec(self) -> ModelSpec | None:
        