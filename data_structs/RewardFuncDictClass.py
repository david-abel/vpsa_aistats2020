# Python imports.
import random
from collections import defaultdict

class RewardFuncDict(object):

    def __init__(self, reward_func_lambda, state_space, action_space, sample_rate=10):
        '''
        Args:
            reward_func_lambda (lambda : simple_rl.State x str --> float)
            state_space (list)
            action_space (list)
            sample_rate (int)
        '''
        self.reward_dict = make_rew_dict_from_lambda(reward_func_lambda, state_space, action_space, sample_rate)

    def reward_func(self, state, action, next_state):
        '''
        Args:
            state (simple_rl.State)
            action (str)

        Returns:
            (float)
        '''
        return self.reward_dict[state][action][next_state]


def make_rew_dict_from_lambda(reward_func_lambda, state_space, action_space, sample_rate):
    '''
    Args:
        transition_func_lambda (lambda : simple_rl.State x str --> simple_rl.State)
        state_space (list)
        action_space (list)
        sample_rate (int)

    Returns:
        (dict)
    '''
    reward_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(float)))
    for s in state_space:
        for a in action_space:
            for s_prime in state_space:
                for i in range(sample_rate):
                    reward_dict[s][a][s_prime] = reward_func_lambda(s, a, s_prime) / sample_rate

    return reward_dict
