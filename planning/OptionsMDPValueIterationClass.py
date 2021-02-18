# Python imports.
from __future__ import print_function
from collections import defaultdict
import random

# Check python version for queue module.
import sys
if sys.version_info[0] < 3:
	import Queue as queue
else:
	import queue

from simple_rl.planning.ValueIterationClass import ValueIteration

class OptionsMDPValueIteration(ValueIteration):
    '''Value iteration class for use with the OptionsMDP, in which not all actions are always available'''

    def __init__(self, mdp, name="value_iter", delta=0.0001, max_iterations=500, sample_rate=3):
        ValueIteration.__init__(self, mdp, name, delta, max_iterations, sample_rate)
        # Including for clarity. OptionsMDPValueIteration gets actions from its
        # MDP instance, and not from the self.actions variable in the Planner class.
        self.actions = None

    def _compute_matrix_from_trans_func(self):
        if self.has_computed_matrix:
            self._compute_reachable_state_space()
            # We've already run this, just return.
            return

            # K: state
                # K: a
                    # K: s_prime
                    # V: prob

        for s in self.get_states():
            for a in self.mdp.get_actions(s):
                for sample in range(self.sample_rate):
                    s_prime = self.transition_func(s, a)
                    self.trans_dict[s][a][s_prime] += 1.0 / self.sample_rate

        self.has_computed_matrix = True

    def _compute_reachable_state_space(self):
        '''
        Summary:
            Starting with @self.start_state, determines all reachable states
            and stores them in self.states.
        '''

        if self.reachability_done:
            return

        state_queue = queue.Queue()
        state_queue.put(self.init_state)
        self.states.add(self.init_state)

        while not state_queue.empty():
            s = state_queue.get()
            for a in self.mdp.get_actions(s):
                for samples in range(self.sample_rate): # Take @sample_rate samples to estimate E[V]
                    next_state = self.transition_func(s,a)

                    if next_state not in self.states:
                        self.states.add(next_state)
                        state_queue.put(next_state)

        self.reachability_done = True

    def run_vi(self):
        '''
        Returns:
            (tuple):
                1. (int): num iterations taken.
                2. (float): value.
        Summary:
            Runs ValueIteration and fills in the self.value_func.
        '''
        # Algorithm bookkeeping params.
        iterations = 0
        max_diff = float("inf")
        self._compute_matrix_from_trans_func()
        state_space = self.get_states()
        self.bellman_backups = 0

        # Main loop.
        while max_diff > self.delta and iterations < self.max_iterations:
            max_diff = 0
            for s in state_space:
                self.bellman_backups += 1
                if s.is_terminal():
                    continue

                max_q = float("-inf")
                for a in self.mdp.get_actions(s):
                    q_s_a = self.get_q_value(s, a)
                    max_q = q_s_a if q_s_a > max_q else max_q

                # Check terminating condition.
                max_diff = max(abs(self.value_func[s] - max_q), max_diff)

                # Update value.
                self.value_func[s] = max_q
            iterations += 1

        value_of_init_state = self._compute_max_qval_action_pair(self.init_state)[0]
        self.has_planned = True

        return iterations, value_of_init_state

    def get_max_q_actions(self, state):
        '''
        Args:
            state (State)
        Returns:
            (list): List of actions with the max q value in the given @state.
        '''
        max_q_val = self.get_value(state)
        best_action_list = []

        # Find best action (action w/ current max predicted Q value)
        for action in self.mdp.get_actions(state):
            q_s_a = self.get_q_value(state, action)
            if q_s_a == max_q_val:
                best_action_list.append(action)

        return best_action_list

    def _compute_max_qval_action_pair(self, state):
        '''
        Args:
            state (State)
        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        '''
        # Grab random initial action in case all equal
        max_q_val = float("-inf")
        if len(self.mdp.get_actions(state)) == 0:
            print("OptionsMDPVIclass", state)
        best_action = self.mdp.get_actions(state)[0]

        # Find best action (action w/ current max predicted Q value)
        for action in self.mdp.get_actions(state):
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action

        return max_q_val, best_action
