# Python imports.
import random
import numpy as np
from collections import defaultdict

class TransitionFuncDict(object):

    def __init__(self, transition_func_lambda, state_space, action_space, sample_rate=1):
        '''
        Args:
            transition_func_lambda (lambda : simple_rl.State x str --> simple_rl.State)
            state_space (list)
            action_space (list)
            sample_rate (int)
        '''
        self.transition_dict = make_trans_dict_from_lambda(transition_func_lambda, state_space, action_space, sample_rate)

    def transition_func(self, state, action):
        '''
        Args:
            state (simple_rl.State)
            action (str)
        '''
        next_state_sample_list = list(np.random.multinomial(1, self.transition_dict[state][action].values()).tolist())
        if len(self.transition_dict[state][action].keys()) == 0:
            return state
        return self.transition_dict[state][action].keys()[next_state_sample_list.index(1)]


def make_trans_dict_from_lambda(transition_func_lambda, state_space, action_space, sample_rate=1):
    '''
    Args:
        transition_func_lambda (lambda : simple_rl.State x str --> simple_rl.State)
        state_space (list)
        action_space (list)
        sample_rate (int)

    Returns:
        (dict)
    '''
    transition_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    
    for s in list(state_space)[:]:
        for a in action_space:
            for i in range(sample_rate):
                s_prime = transition_func_lambda(s, a)

                transition_dict[s][a][s_prime] +=  (1.0 / sample_rate)

    return transition_dict
