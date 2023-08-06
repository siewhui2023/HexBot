import sys
import time
import math
from statistics import mean
import numpy as np
from state import State
from constants import *
from environment import *
"""
solution.py
"""

### REFERENCES #########################################################################################################
## 
## 1. COMP3702 Tutorial 9: GridWorld_RL_soln.py
##
## 2. Reinforcement Learning: An Introduction (2nd Edition) by Richard S. Sutton and Andrew G. Barto
##
## 3. Github from COMP3702 Tutor: Peter
##    https://github.com/comp3702/tutorial10
##
## 4. There is one part of the codes (specific mentioned below) are given from my tutor during the tutorial class 
##
########################################################################################################################

class RLAgent:

    #
    # (optional) Define any constants you require here.
    #

    def __init__(self, environment: Environment):
        self.environment = environment

        self.exploit_prob = 0.5                          # 'epsilon' in epsilon-greedy
        # self.learning_rate = self.environment.alpha      # 'alpha' in lecture notes
        # self.learning_rate = 0.002
        # self.learning_rate = 0.02
        self.learning_rate = 0.2
        self.gamma = self.environment.gamma              # 'gamma' in lecture notes

        self.init_state = self.environment.get_init_state()
        self.init_s = {self.init_state}

        self.q_values = {}
        self.one = {}
        self.two = {}

        self.start = time.time()

        self.unvisited = []
        self.exits = False 
    

    # === Q-learning ===================================================================================================

    def q_learn_train(self):
        """
        Train this RL agent via Q-Learning.
        """
        #
        # TODO: Implement your Q-learning training loop here.
        #

        f = open("Untitled1.txt", "w")

        episode = 0

        while self.environment.get_total_reward() > self.environment.training_reward_tgt and \
                self.environment.training_time_tgt - 1 > time.time() - self.start:

            states = random.choice(list(self.init_s))

            for n in range(50):
                self.exits = False

                action = self.get_best_action(states)
                self.adding(states, action)
                # next_state, solved = self.next_iteration(states, action)

                reward, next_state = self.environment.perform_action(states, action)
                solved = self.environment.is_solved(next_state)

                next_q = float('-inf')
                next_a = None

                # Update q-value for the (state, action) pair
                old_q = self.q_values.get((states, action), 0)

                for actionss in ROBOT_ACTIONS:
                    q = self.q_values.get((next_state, actionss), 0)
                    if q is not None and q > next_q:
                        next_q = q
                        next_a = actionss
                if next_a is None:
                    next_q = 0

                target = reward + (self.gamma * next_q)

                new_q = old_q + (self.learning_rate * (target - old_q))
                self.q_values[(states, action)] = new_q
        
                if solved:
                    break

                states = next_state
                self.init_s.add(states)

            episode += 1  
        
        
            f.write(str(reward) + "\n")
            

    def q_learn_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via Q-learning.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  Q-learning Q-values) here.
        #
        best_q = float('-inf')
        best_a = None
        for acts in ROBOT_ACTIONS:
            if (state, acts) not in self.q_values:
                self.unvisited.append(acts)
                self.exits = True
            elif not self.exits:    
                q = self.q_values[(state, acts)]
                if q is not None and q > best_q:
                    best_q = self.q_values[(state, acts)]
                    best_a = acts
        if self.exits:
            return random.choice(self.unvisited)
        else:
            return best_a
        

    # === SARSA ========================================================================================================

    def sarsa_train(self):
        """
        Train this RL agent via SARSA.
        """
        #
        # Implement your SARSA training loop here.
        #
        episode = 0

        while self.environment.get_total_reward() > self.environment.training_reward_tgt and \
                self.environment.training_time_tgt - 1 > time.time() - self.start:

            states = random.choice(list(self.init_s))

            for n in range(50):
                self.exits = False

                action = self.get_best_action(states)
                self.adding(states, action)
                next_state, solved = self.next_iteration(states, action)

                if solved:
                    break

                states = next_state
                # actions = next_action
                self.init_s.add(states)

            episode += 1

    def sarsa_select_action(self, state: State):
        """
        Select an action to perform based on the values learned from training via SARSA.
        :param state: the current state
        :return: approximately optimal action for the given state
        """
        #
        # TODO: Implement code to return an approximately optimal action for the given state (based on your learned
        #  SARSA Q-values) here.
        #
        best_q = float('-inf')
        best_a = None
        for acts in ROBOT_ACTIONS:
            if (state, acts) not in self.q_values:
                self.unvisited.append(acts)
                self.exits = True
            elif not self.exits:    
                q = self.q_values[(state, acts)]
                if q is not None and q > best_q:
                    best_q = self.q_values[(state, acts)]
                    best_a = acts
        if self.exits:
            return random.choice(self.unvisited)
        else:
            return best_a

    # === Helper Methods ===============================================================================================
    #
    # (optional) Add any additional methods here.
    #

    ## This part of codes are the hints that I got from the tutorial class
    def get_best_action(self, states):
        best_u = float('-inf')
        best_a = None

        for act in ROBOT_ACTIONS:
            if (states, act) not in self.q_values:
                self.unvisited.append(act)
                self.exits = True
            elif not self.exits:
                u = self.q_values[(states, act)] + (self.exploit_prob * math.sqrt(math.log(self.one[states])/ self.two[states, act]))
                if u > best_u:
                    best_u = u
                    best_a = act

        if self.exits:
            action = random.choice(self.unvisited)
        else:
            action = best_a
            
        return action


    ## Mainly from COMP3702 Tutorial 9: GridWorld_RL_soln.py
    def next_iteration(self, states, action):

        # Choose an action, simulate it, and receive a reward
        reward, next_state = self.environment.perform_action(states, action)
        solved = self.environment.is_solved(next_state)

        next_q = float('-inf')
        next_a = None

        # Update q-value for the (state, action) pair
        old_q = self.q_values.get((states, action), 0)

        for actionss in ROBOT_ACTIONS:
            q = self.q_values.get((next_state, actionss), 0)
            if q is not None and q > next_q:
                next_q = q
                next_a = actionss
        if next_a is None:
            next_q = 0

        target = reward + (self.gamma * next_q)

        new_q = old_q + (self.learning_rate * (target - old_q))
        self.q_values[(states, action)] = new_q
        

        return next_state, solved


    def adding(self, states, action):

        if states in self.one:
            self.one[states] += 1
        else:
            self.one[states] = 1
            
        if (states, action) in self.two:
            self.two[states, action] += 1
        else:
            self.two[states, action] = 1

