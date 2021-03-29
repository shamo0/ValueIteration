import json
import numpy as np
import copy
import operator

class World():

    def __init__(self, inputFileName):

        with open(inputFileName, 'r') as filehandle:
            item = json.load(filehandle)
        self.shape = item['shape']
        self.gamma = item['gamma']
        self.rl = item['rl']
        self.tl = item['tl']
        self.bl = item['bl']
        self.values = [[0 for j in range(self.shape[1])] for i in range(self.shape[0])] #Initialize values to zero
        self.policy = [['N' for j in range(self.shape[1])] for i in range(self.shape[0])] #Initialize policy to N

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if [i,j] in self.tl:
                    self.policy[i][j] = '.'
                elif [i,j] in self.bl:
                    self.policy[i][j] = '.'

        
   
        self.current_state = [0,0]
        self.new_state = [0,0]

        self.actions = ["N","E","W","S"]

    def move(self,direction):
        move_reward = 0 
        if direction == "N":
            if self.current_state[0]-1 == -1:
                self.new_state = self.current_state
            elif self.policy[self.current_state[0]-1][self.current_state[1]] ==".":
                for i in self.rl:
                    if i[0] == [self.current_state[0]-1,self.current_state[1]]:
                        move_reward = i[1]
                        self.new_state = [self.current_state[0]-1,self.current_state[1]]
                        return self.new_state, move_reward
                    else:
                        self.new_state = self.current_state
            else:
                self.new_state = [self.current_state[0]-1,self.current_state[1]]
        elif direction == "E":
            if self.current_state[1]+1 == self.shape[1]:
                self.new_state = self.current_state
            elif self.policy[self.current_state[0]][self.current_state[1]+1] ==".":
                for i in self.rl:
                    if i[0]== [self.current_state[0],self.current_state[1]+1]:
                        move_reward = i[1]
                        self.new_state = [self.current_state[0],self.current_state[1]+1]
                        return self.new_state, move_reward
                    else:
                        self.new_state = self.current_state
            else:
                self.new_state = [self.current_state[0],self.current_state[1]+1]
        elif direction == "S":
            if self.current_state[0]+1 == self.shape[0]:
                self.new_state = self.current_state
            elif self.policy[self.current_state[0]+1][self.current_state[1]] ==".":
                for i in self.rl:
                    if i[0]== [self.current_state[0]+1,self.current_state[1]]:
                        move_reward = i[1]
                        self.new_state = [self.current_state[0]+1,self.current_state[1]]
                        return self.new_state, move_reward
                    else:
                        self.new_state = self.current_state
            else:
                self.new_state = [self.current_state[0]+1,self.current_state[1]]
        elif direction == "W":
            if self.current_state[1]-1 == -1:
                self.new_state = self.current_state
            elif self.policy[self.current_state[0]][self.current_state[1]-1] ==".":
                for i in self.rl:
                    if i[0] == [self.current_state[0],self.current_state[1]-1]:
                        move_reward = i[1]
                        self.new_state = [self.current_state[0],self.current_state[1]-1]
                        return self.new_state, move_reward
                    else:
                        self.new_state = self.current_state
            else:
                self.new_state = [self.current_state[0],self.current_state[1]-1]
        elif direction == ".":
            pass
        return self.new_state, move_reward


    def possible_actions(self,action):
        maybe_actions = '.'
        if action == "N":
            maybe_actions = ["W","E"]
        elif action == "E":
            maybe_actions = ["N","S"]
        elif action == "S":
            maybe_actions = ["E","W"]
        elif action == "W":
            maybe_actions = ["S","N"]
        return maybe_actions

    def valueIteration(self,num_iterations):
        while num_iterations:
            updated_values = copy.deepcopy(self.values)
            updated_policy = copy.deepcopy(self.policy)

            for m in range(int(self.shape[0])):
                for n in range(int(self.shape[1])):
                    action_policy = {'N':-1,"E":-1,"S":-1,"W":-1}

                    # self.current_state = [m,(self.shape[1]-1)-n]
                    self.current_state = [m,n]

                    # Check if current state blocked or terminal
                    if self.policy[m][n] =='.':
                        continue
                    
                    old_v = self.values[self.current_state[0]][self.current_state[1]]
                    

                    for action in self.actions:
                        possible_v = {}
                        
                        possibles = self.possible_actions(action)
                        left = possibles[0]
                        right = possibles[1]

                        actual_new_state,actual_reward = self.move(action)
                        left_new_state,left_reward = self.move(left)
                        right_new_state,right_reward = self.move(right)

                        allRewards = [actual_reward,left_reward,right_reward]
                        allActions = [actual_new_state,left_new_state,right_new_state]   
                        v = 0.8*(actual_reward+self.gamma*self.values[actual_new_state[0]][actual_new_state[1]]) + 0.1*(left_reward+self.gamma*self.values[left_new_state[0]][left_new_state[1]]) + 0.1*(right_reward+self.gamma*self.values[right_new_state[0]][right_new_state[1]]) 

                        possible_v[action] = v

                        best_policy = 'N'

                        for k in possible_v:
                            if possible_v[k] > action_policy[k]:
                                action_policy[k] = possible_v[k]

                            else:
                                continue
                        best_policy = max(action_policy.items(),key=operator.itemgetter(1))[0]
                        updated_policy[self.current_state[0]][self.current_state[1]] = best_policy
                        best_value = action_policy[best_policy]
                    updated_values[self.current_state[0]][self.current_state[1]] = best_value

            self.policy = updated_policy
            self.values = updated_values
            num_iterations -=1

