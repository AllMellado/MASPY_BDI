
class LearningMultiton(type):
    _instances: Dict[str, "Learning"] = {}
    _lock: Lock = Lock()

    def __call__(cls, lrn_name=None, full_log=None):
        with cls._lock:
            _my_name = lrn_name if lrn_name else str(cls.__name__)
            if _my_name not in cls._instances:
                vars = []
                if lrn_name: vars.append(lrn_name)
                if lrn_name: vars.append(full_log)
                instance = super().__call__(*vars)
                cls._instances[_my_name] = instance
        return cls._instances[_my_name]

class Learning(metaclass=LearningMultiton):
    def __init__(self, lrn_name=None,full_log=None) -> None:
        
        self._my_name = lrn_name
        self.full_log = full_log
        self.model = Model 
        #self.map_size = map_size
        #self.num_targets = num_targets
        #self.wall_perc = wall_perc
        #self.epsilon = epsilon
        #self.num_pop = num_pop
        #self.num_step = num_step
        #self.num_iter = num_iter
        
        self.agent_list = dict()
        self._agents = dict()
        
        self._name = f"Learning:{self._my_name}"
        self.make_graph = False
        self.adj_matrix = None
        self.frames = None
        self.methods = [self.q_learning, self.sarsa]
        self.policy = None
    
    def print(self,*args, **kwargs):
        return print(f"{self._name}>",*args,**kwargs)
    
    def strategy(self):
        return self.policy
    
    def add_agents(self, agents):
        try:
            for agent in agents:
                self._add_agent(agent)
        except TypeError:
            self._add_agent(agents)
    
    def _add_agent(self, agent):
        from maspy.agent import Agent
        assert isinstance(agent,Agent)
        
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
        
        self.print(f'Connecting agent {type(agent).__name__}:{agent.my_name}') if self.full_log else ...
        
    def set_params(self, map_size=10, num_targets=10, wall_perc=0.1,
                epsilon=0.01, num_pop=5, num_step=50, num_iter=100):
        self.map_size = int(map_size)
        self.num_targets = int(num_targets)
        self.wall_perc = wall_perc
        self.epsilon = epsilon
        self.num_pop = int(num_pop)
        self.num_step = int(num_step)
        self.num_iter = int(num_iter)
    
    def set_env(self, map):
        self.env = self.model(map)
        size = self.env.num_rows*self.env.num_cols
        self.adj_matrix = np.zeros((size, size), dtype=int) 
        self.frames = []
    
    def get_action_reward(self, coordinates: tuple):
        state = self.env.encode(*coordinates)
        actions = self.policy[state]
        action =  np.argmax(actions)
        self.env.set_state(state)
        s, r, t, d = self.env.look(action)
        return action, r
    
    def normalize(self, numbers):
        min_val = min(numbers)
        max_val = max(numbers)
        total = sum(numbers)
        range_value = max_val - min_val
        if range_value == 0:
            return [0 for num in numbers]
        else:
            return [(num - min_val)/total for num in numbers]

    def monte_carlo_selection(self, probabilities):
        prob_aux = self.normalize(probabilities)
    
        rand_num = np.random.uniform(0, 1)
    
        position = 0
        for i, p in enumerate(prob_aux):
            if rand_num <= p:
                position = i
                break
            else:
                rand_num -= p
    
        return position

    def shortest_path(self, adj_matrix,start_state,q_table=[],end_state=None):
        n = len(adj_matrix)
        
        seen = [False]*n
        seen[start_state] = True
        distances = [-1]*n
        distances[start_state] = 0
        parent = [-1]*n

        queue = []
        queue.append(start_state)

        closest_state = -1
        best_value = -1

        while queue:
            state = queue.pop(0)
            seen_flag = False
            for vertex, edge in enumerate(adj_matrix[state]):
                if edge != 0 and not seen[vertex] and adj_matrix[vertex,vertex] != 3:
                    seen_flag = True
                    seen[vertex] = True
                    distances[vertex] = distances[state] + 1
                    parent[vertex] = state
                    queue.append(vertex)
                    if adj_matrix[vertex,vertex] < 2:
                        try:
                            vertex_value = (max(q_table[vertex])*4)/distances[vertex]
                            if vertex_value > best_value:
                                closest_state = vertex
                                best_value = vertex_value
                        except:
                            closest_state = vertex
                            queue.clear()
        
        if closest_state != -1:
            path = []
            current = closest_state
            while current != start_state:
                path.append(current)
                current = parent[current]

            return list(reversed(path))
        else:    
            print(f'Return None for shortest path: {start_state} {state} {closest_state} ')
            print(f'Submatrix:\n{self.submatrix(adj_matrix,[state,state],1)}')
            return None

    def show_frames(self, frames=None):
        if not frames:
            frames = self.frames
        for i, frame in enumerate(frames):
            system('cls')
            print(frame['frame'])
            print(f"Timestep: {i + 1} of {len(frames)}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Action Flag: {frame['action_flag']}")
            print(f"Best Actions: {frame['best_actions']}")
            print(f"Possible Actions: {frame['possible_actions']}")
            print(f"Reward: {frame['reward']}")
            print(f"Acumulated Reward: {frame['acumulated_reward']}")
            sys.stdout.flush()
            sleep(0.8)

    def random_map(self):
        size = self.map_size
        if self.wall_perc > 0:
            wall_num = int(self.wall_perc*size*size) + np.random.randint(int(self.wall_perc*size*size))
        else: 
            wall_num = 0
        wall_aux = wall_num
        dirt_num = self.num_targets
        dirt_aux = dirt_num

        #print(f'Making {(size,size)} map with {wall_num} walls and {dirt_num} dirts')
        total_path = []
        while len(total_path) < size*size - wall_num + 1:
            map = np.zeros([size, size])
            done = False
            while not done:
                if wall_num > 0:
                    x = np.random.randint(size-1)
                    y = np.random.randint(size-1)
                    if map[x,y] == 0:
                        map[x,y] = -1
                        wall_num -= 1

                if dirt_num > 0:
                    x = np.random.randint(size-1)
                    y = np.random.randint(size-1)
                    if map[x,y] == 0:
                        map[x,y] = -2
                        dirt_num -= 1

                if wall_num == 0 and dirt_num == 0:
                    done = True
            
            #Checking map traversability
            wall_num = wall_aux
            dirt_num = dirt_aux
            coord = np.random.randint(low=0,high=size-1,size=2)
            graph = [coord]
            total_path = []
            seen = np.zeros((size,size),dtype=int)
            while graph:
                pos = graph.pop(0)
                total_path.append(pos)
                for i in range(max(0,pos[0]-1),min(size-1,pos[0]+1)+1):
                    if map[i,pos[1]] != -1 and seen[i,pos[1]] == 0:
                        graph.append([i,pos[1]])
                        seen[i,pos[1]] = 1
                for j in range(max(0,pos[1]-1),min(size-1,pos[1]+1)+1):
                    if map[pos[0],j] != -1 and seen[pos[0],j] == 0:
                        graph.append([pos[0],j])
                        seen[pos[0],j] = 1
        try:
            return map
        except Exception as e:
            print(f"{len(total_path) < size*size-wall_num} : {e}")
            sys.exit()

    def submatrix(self, matrix,center,radius):
        x, y = center
        subm_rows = slice(max(0, x-radius), min(x+radius+1, matrix.shape[0]))
        subm_cols = slice(max(0, y-radius), min(y+radius+1, matrix.shape[1]))
        subm = np.zeros((2*radius+1, 2*radius+1), dtype=matrix.dtype)
        subm[:] = -1
        subm[radius-x+subm_rows.start:radius-x+subm_rows.stop, 
            radius-y+subm_cols.start:radius-y+subm_cols.stop] = matrix[subm_rows, subm_cols]
        return subm

    def multi_reward_log(self, max, x, factor=10):
        try:
            value = ((1-factor)/math.log(max))*math.log(x)+factor
        except ValueError:
            print(f'This: {max} - {x} - {factor} resulted in error')
            print(f'Error: {(1-factor)}/{math.log(max)}*{math.log(x)}+{factor}')
        return value

    def all_targets_reached(self, e):
        for col in e.desc:
            if -2 in col:
                return False
        return True
    
    def q_learning(self, q_table, epsilon, e, num_steps=50, num_iter=500,show_log=False):
        assert isinstance(e,Model)
        
        alpha = 0.1
        gamma = 0.95 
        sum_acumulated_reward = []
        convergence_threshold = 0.001
        prev_Q = np.copy(q_table)
        size = e.num_states
        max_steps_reward = size
        # Matrix de adjacencia do grafo de exploracao  
        for idx in range(1,num_iter+1):
            adj_matrix = np.zeros((size, size), dtype=int)
            state, _ = e.reset()    # Resetando aspirador para um estado aleatorio
            acumulated_reward = 0

            adj_matrix[state,state] = 2 # Este valor indica na matrix que este estado ja foi explorado
            path_actions = []
            best_states = []
            steps = 0
            last_step = 0
            while steps < num_steps:
                action, possible_states, path_actions = self.choose_action(e, epsilon, state, q_table, adj_matrix, path_actions)
                next_state, reward,_,_ = e.step(action)
                if reward > 0:
                    reward = reward*self.multi_reward_log(max_steps_reward,steps-last_step+1,3)
                    last_step = steps
                acumulated_reward += int(reward)
                
                if next_state in best_states:
                    best_states.remove(next_state)
                
                # Ajustes na matrix de adjacencia
                # Inserindo ligacoes entre estados (valor 1) e estados esplorados (valor 2)
                adj_matrix[next_state,next_state] = 2
                for p_state in possible_states:
                    if adj_matrix[state,p_state] == 0:
                    # print(f"Connecting {state} to {p_state} -> {next_state} ({possible_states}) {possible_actions}")
                        if adj_matrix[p_state,p_state] == 0:
                            adj_matrix[p_state,p_state] = 1
                        adj_matrix[state,p_state] = 1
                        adj_matrix[p_state,state] = 1

                # Linha onde ocorre a retencao do aprendizado
                q_table[state, action] = q_table[state, action] + alpha*(reward + gamma * np.max(q_table[next_state]) - q_table[state, action])
                state = next_state
                
                #if acumulated_reward >= 10*self.num_targets:
                    #break
                if self.all_targets_reached(e): break
                steps += 1
                
            sum_acumulated_reward.append(acumulated_reward)
            if show_log and idx == 1:
                print(f"{acumulated_reward}",end=" >> ")
                sys.stdout.flush()
                
            if idx > 10:
                delta = np.abs(q_table - prev_Q).max()
                if idx%1000 == 0: print(f'{idx}> Delta: {delta}')
                if delta < convergence_threshold:
                    break
            prev_Q = np.copy(q_table)
        
        if show_log:
            print(f"{np.mean(sum_acumulated_reward,dtype=int)} | idx[{idx}]",end=" | ")
            sys.stdout.flush()
        
        return q_table

    def choose_action(self, e, epsilon, state, table, adj_matrix, path_actions):
        assert isinstance(e,Model)
        
        possible_actions = e.action_mask(state)
        possible_states = e.actions_to_states(possible_actions)
        _, best_actions = e.sensor()
        if sum(best_actions) > 0:   # Usado quando o sensor encontra uma sujeira
            q_actions = np.multiply(table[state],best_actions)
            q_actions = np.add(q_actions,best_actions)
            action = np.argmax(q_actions)
        else:                       
            for ia, act in enumerate(possible_actions): # Verificar disponibilidade de acoes
                if act == 1:
                    s,_,_,_ = e.look(ia)
                    if adj_matrix[s,s] > 1 and ia != 4:
                        possible_actions[ia] = 0
                
            if sum(possible_actions) == 0 or len(path_actions) > 0: # Usado quando aspirador esta perdido
                if len(path_actions) == 0 or path_actions is None:
                    path = self.shortest_path(adj_matrix,state,table)
                    if path is not None:
                        path_actions = e.states_to_actions(path)
                    if path_actions is None or path is None:
                        print(
                            f"\nCurrent: {state} > ({path}) ({path_actions}) \nPossible States:\
                            \n{[(x,i) for x,i in enumerate(adj_matrix[state]) if x in possible_states]}\
                            \nMap Render:\
                            \n{e.render(adj_matrix)}"
                        )
                        sys.exit()
                        
                action = path_actions.pop(0) # Retornar para areas inexploradas
            elif np.random.uniform(0, 1) < epsilon:
                action = self.monte_carlo_selection(possible_actions)  # Explorar diferentes acoes
            else:
                actions = np.multiply(table[state],possible_actions)
                action = np.argmax(actions) # Utilizar valores aprendidos
        return action, possible_states, path_actions

    def sarsa(self, s_table, epsilon, e, num_steps=50, num_iter=500,show_log=False):
        assert isinstance(e,Model)
        
        alpha = 0.1
        gamma = 0.95
        sum_acumulated_reward = []
        size = e.num_states
        for idx in range(1,num_iter+1):
            adj_matrix = np.zeros((size, size), dtype=int) # Matrix de adjacencia do grafo de exploracao  
            state, _ = e.reset()    # Resetando aspirador para um estado aleatorio
            acumulated_reward = 0

            adj_matrix[state,state] = 2 # Este valor indica na matrix que este estado ja foi explorado
            path_actions = []
            
            action,possible_states,path_actions = self.choose_action(e,epsilon,state,s_table,adj_matrix,path_actions)
            steps = 0
            while steps < num_steps:
                next_state, reward,_,_ = e.step(action)
                acumulated_reward += reward
                
                # Ajustes na matrix de adjacencia
                # Inserindo ligacoes entre estados (valor 1) e estados esplorados (valor 2)
                adj_matrix[next_state,next_state] = 2
                for p_state in possible_states:
                    if adj_matrix[state,p_state] == 0:
                    # print(f"Connecting {state} to {p_state} -> {next_state} ({possible_states}) {possible_actions}")
                        if adj_matrix[p_state,p_state] == 0:
                            adj_matrix[p_state,p_state] = 1
                        adj_matrix[state,p_state] = 1
                        adj_matrix[p_state,state] = 1
                        
                next_action, possible_states, path_actions = self.choose_action(e,epsilon,next_state,s_table,adj_matrix,path_actions)

                # Linha onde ocorre a retencao do aprendizado
                
                s_table[state, action] = s_table[state, action]+ alpha*(reward + gamma * s_table[next_state][next_action] - s_table[state, action])
                
                state = next_state
                action = next_action
                steps += 1

            sum_acumulated_reward.append(acumulated_reward)
            if show_log:
                if idx == 1:
                    print(f"{acumulated_reward}",end=" >> ")
                    sys.stdout.flush()
                if idx == num_iter:
                    print(f"{np.mean(sum_acumulated_reward,dtype=int)}",end=" | ")
                    sys.stdout.flush()
        return s_table

    def learn(self, method_index=0, show_log=False):
        print(f"Learning with {self.methods[method_index].__name__}...")
        print(f'  Average Reward for {self.num_step} steps\n  First and Last Iteration') if show_log else ...
        table = np.zeros([self.map_size**2, self.model().num_actions])
        method = self.methods[method_index]
        start_time = time.time()
        for i in range(1, self.num_pop+1):
            print(f'Map {i}: ',end='') if show_log else ...
            map = self.random_map()
            env = self.model(map)
            table = method(table, self.epsilon, env, self.num_step, self.num_iter,show_log) 
            print('') if show_log else ...
        print(f"Finished in {round(time.time()-start_time,4)} seconds!")
        self.policy = table
        
    def exec(self, env, state, num_steps=1, show=False):
        if self.policy is None:
            self.policy = np.zeros([self.map_size**2, self.model().num_actions])
            
        if self.adj_matrix is None: 
            size = env.num_rows*env.num_cols
            self.adj_matrix = np.zeros((size, size), dtype=int) 
        
        env.set_state(state)
        #state, _ = env.encode(x,y)
        reward = 0
        acumulated_reward = 0
        actions_taken = []
        states_arrived = []
        
        self.adj_matrix[state,state] = 2
        possible_actions = []
        path_actions = []
        frames = []
        for _ in range(num_steps):
            flag = 'Best'     
            possible_actions = env.action_mask(state)
            possible_states = env.actions_to_states(possible_actions)
            _, best_actions = env.sensor()
            if sum(best_actions) > 0:
                actions = np.multiply(self.policy[state],best_actions)
                actions = np.add(actions,best_actions)
                action = np.argmax(actions)
            else:
                for ia, act in enumerate(possible_actions):
                    if act == 1:
                        s,_,_,_ = env.look(ia)
                        if self.adj_matrix[s,s] > 1 and ia != 4:
                            possible_actions[ia] = 0
                actions = np.multiply(self.policy[state],possible_actions)
                if sum(actions) == 0 and sum(possible_actions) > 0:
                    actions = possible_actions
                if sum(actions) > 0 and len(path_actions) == 0:
                    flag = 'Policy'
                    action = np.argmax(actions)
                else:
                    flag = 'Return'
                    if len(path_actions) == 0:
                        path = self.shortest_path(self.adj_matrix,state,self.policy)
                        if path is not None:
                            path_actions = env.states_to_actions(path)
                        if path_actions is None or path is None:
                            print(
                                f"\nCurrent: {state} > ({path}) ({path_actions})\
                                \nAvailable Actions: {actions} = {self.policy[state]}*{possible_actions}\
                                \nPossible States:\
                                \n{[(x,i) for x,i in enumerate(self.adj_matrix[state]) if x in possible_states]}\
                                \nMap Render:\
                                \n{env.render(self.adj_matrix)}"
                            )
                            sys.exit()
                    action = path_actions.pop(0)

            self.frames.append({
                'frame': env.render(self.adj_matrix),
                'state': state,
                'action': env.last_action,
                'action_flag': flag,
                'best_actions': best_actions,
                'possible_actions': possible_actions,
                'reward': reward,
                'acumulated_reward': acumulated_reward,
            })

            actions_taken.append(action)
            next_state, reward,_,_ = env.step(action)
            states_arrived.append(next_state)

            self.adj_matrix[next_state,next_state] = 2
            for p_state in possible_states:
                if self.adj_matrix[state,p_state] == 0:
                    if self.adj_matrix[p_state,p_state] == 0:
                        self.adj_matrix[p_state,p_state] = 1
                    self.adj_matrix[state,p_state] = 1
                    self.adj_matrix[p_state,state] = 1

            state = next_state
            acumulated_reward += reward
        
        if show:
            self.show_frames(frames)
        if num_steps == 1:
            return actions_taken[0], states_arrived[0], acumulated_reward
        return actions_taken, states_arrived, acumulated_reward
        
        

        self.policy