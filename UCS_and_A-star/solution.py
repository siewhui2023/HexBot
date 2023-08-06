from operator import lt
import sys
import heapq
import math
from constants import *
from environment import *
from state import State
"""
solution.py

"""


class Solver:

    def __init__(self, environment, loop_counter):
        self.environment = environment
        self.loop_counter = loop_counter
        self.visited = set()


    # === Reference ================================================================================================
    # COMP3702 Tutorial 2 solutions
    # File_name: Tutorial_2_solutions.ipnyb
    #
    #
    # AND
    #
    # COMP3702 Tutorial 3 solutions
    # File_name: tutorial3.py
    #
    # ==============================================================================================================
    def solve_ucs(self):
        print("Solving UCS, please wait...")

        initial_state =  Node(self.environment.get_init_state(), 0, [])
        self.visited = {initial_state.states: 0}

        pq = []
        pq.append(initial_state)
        heapq.heapify(pq)
        
        
        while (len(pq) > 0):
            current_node = heapq.heappop(pq)
            self.loop_counter.inc()
            
            if self.environment.is_solved(current_node.states):
                print("We reached the goal, Congratulation!") 
                print(len(self.visited))
                return current_node.actions

            for action in ROBOT_ACTIONS:
                succ, cost, new_state = self.environment.perform_action(current_node.states, action)
                if (succ == False):
                    continue 
                cost += current_node.sub_cost

                if (new_state not in self.visited.keys()) or (cost < self.visited[new_state]):
                    self.visited[new_state] = cost
                    latest_node = Node(new_state, cost, current_node.actions + [action])
                    heapq.heappush(pq, latest_node)

        if not pq:
            current_node = None  

        return current_node, self.visited



    # === Reference ================================================================================================
    # COMP3702 Tutorial 2 solutions
    # File_name: Tutorial_2_solutions.ipnyb
    #
    # AND
    #
    # COMP3702 Tutorial 3 solutions (provided by my tutor, Mr Vektor Dewanto)
    # Link: https://gist.github.com/tttor/826be15b99bb4b33a50787d7eb7b5fda
    #
    # AND
    #
    # COMP3702 Tutorial 3 solutions
    # File_name: tutorial3.py
    # 
    # ==============================================================================================================
    def solve_a_star(self):
        print("Solving A*, please wait...")

        initial_state =  Node(self.environment.get_init_state(), 0, [])
        self.visited = {initial_state.states: 0}

        pq = []
        pq.append(initial_state)
        heapq.heapify(pq)
        
        while (len(pq) > 0):
            current_node = heapq.heappop(pq)
            self.loop_counter.inc()
           
            if self.environment.is_solved(current_node.states):
                print("We reached the goal, Congratulation!") 
                print(len(self.visited))
                return current_node.actions

            for action in ROBOT_ACTIONS:
                succ, cost, new_state = self.environment.perform_action(current_node.states, action)
                if (succ == False):
                    continue
                cost += current_node.sub_cost + self.estimate_cost_to_go(current_node.states, 'manhattan')

                if (new_state not in self.visited.keys()) or (cost < self.visited[new_state]):
                    self.visited[new_state] = cost
                    latest_node = Node(new_state, cost, current_node.actions + [action])
                    heapq.heappush(pq, latest_node)

        if not pq:
            current_node = None  

        return current_node, self.visited



    def converted_coordinates(self, coordinate):
        # Reference: 
        # Title: Introduction to Axial Coordinates for Hexagonal Tile-Based Games
        # Section 1. Axial Coordinates
        # Author: Juwul Bose.  Date of published: Aug 15, 2017.
        # link: https://gamedevelopment.tutsplus.com/tutorials/introduction-to-axial-coordinates-for-hexagonal-tile-based-games--cms-28820
        x_value = coordinate[0]
        y_value = coordinate[1] - ( math.floor (x_value / 2) )
        return x_value, y_value


    def Path_finding(self, coordinate1, coordinate2):
        # Reference:
        # Title: Computation of Compact Distributions of Discrete Elements
        # Section 3.1 Voronoi Diagram Based on Several Distance Metrics (Pgae 5 of 16)
        # Author: Jie Chen, Gang Yang *, Meng Yang
        # link: https://www.researchgate.net/publication/331203691_Computation_of_Compact_Distributions_of_Discrete_Elements#pf6
        #
        # AND
        #
        # Title: Pathfinding on a hexagonal grid – A* Algorithm
        # Author: Michat Magdziarz
        # link: https://blog.theknightsofunity.com/pathfinding-on-a-hexagonal-grid-a-algorithm/

        # Fromula used: | x1 - x2 | + | y1 - y2 |  ,  from Pathfinding on a hexagonal grid – A* Algorithm

        part_1 = abs(coordinate1[0] - coordinate2[0])
        part_2 = abs(coordinate1[1] - coordinate2[1])
        total = abs(part_1 + part_2) 
        return total


    def estimate_cost_to_go(self, state, heuristic_mode=None):
        # Reference:
        # COMP3702 Tutorial 3 solutions (provided by my tutor, Mr Vektor Dewanto)
        # Link: https://gist.github.com/tttor/826be15b99bb4b33a50787d7eb7b5fda

        cost_to_go_estimate = 0
        
        if heuristic_mode=='zeroed':
            cost_to_go_estimate = 0 
        elif heuristic_mode=='manhattan':
            for i in range(len(state.widget_centres)):
                cost_to_go_estimate += self.Path_finding(self.converted_coordinates(self.environment.target_list[i]), 
                                                        self.converted_coordinates(state.widget_centres[i]))

        else:
            raise NotImplementedError(heuristic_mode)
        return cost_to_go_estimate



# === Reference ================================================================================================
# COMP3702 Tutorial 3 solutions
# File_name: tutorial3.py
#
# ==============================================================================================================
class Node(): 
    def __init__(self, state, cost, actions):
        self.sub_cost = cost
        self.states = state
        self.actions = actions

    def __lt__(self, other): 
        return self.sub_cost < other.sub_cost

