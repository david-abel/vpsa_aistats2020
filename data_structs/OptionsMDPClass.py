from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
from simple_rl.planning.ValueIterationClass import ValueIteration

from planning.OptionsMDPValueIterationClass import OptionsMDPValueIteration

class OptionsMDP(MDP):
    '''An MDP where not all actions are available in all states'''
    def __init__(self, actions, transition_func, reward_func, init_state, gamma=0.99, step_cost=0, str='OMDP'):
        MDP.__init__(self, actions, transition_func, reward_func, init_state, gamma, step_cost)
        self.str = str

    @property
    def actions(self):
        raise NotImplementedError("Can't access actions directly in OptionsMDP."
                                  " Use get_actions or get_all_actions instead.")

    @actions.setter
    def actions(self, val):
        self._actions = val

    def get_actions(self, state=None):
        if state is None:
            state = self.cur_state
        actions = [op for op in self._actions if op.is_init_true(state)]
        if len(actions) == 0:
            print state
        return actions


    def get_all_actions(self):
        return self._actions

    def execute_agent_action(self, action):
        '''
        Args:
            action (str)
        Returns:
            (tuple: <float,State>): reward, State
        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        '''
        if not action.is_init_true(self.cur_state):
            print "(OptionsMDP) Warning: can't initiate option", action.name, "in state", str(self.cur_state)
            return 0, self.cur_state

        reward = self.reward_func(self.cur_state, action)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def __str__(self):
        return self.str

    def __repr__(self):
        return self.str


def mdp_to_options_mdp(mdp):
    options = [primitive_action_to_option(a) for a in mdp.actions]

    def omdp_transition_func(s, o):
        action = o.policy(s)
        return mdp.transition_func(s, action)

    def omdp_reward_func(s, o):
        action = o.policy(s)
        return mdp.reward_func(s, action)

    omdp = OptionsMDP(actions=options,
                      transition_func=omdp_transition_func,
                      reward_func=omdp_reward_func,
                      init_state=mdp.init_state,
                      gamma=mdp.gamma,
                      step_cost=mdp.step_cost,
                      str=str(mdp))

    # Add other properties from base mdp onto the OptionsMDP
    for attr in mdp.__dict__:
        if attr not in omdp.__dict__:
            omdp.__dict__[attr] = mdp.__dict__[attr]

    return omdp

def primitive_action_to_option(action):
    true_predicate = Predicate(lambda s: True)
    policy = lambda s: action

    return Option(init_predicate=true_predicate,
                  term_predicate=true_predicate,
                  policy=policy,
                  name='o_' + str(action))


if __name__ == '__main__':

    s0 = State(data=0)
    s1 = State(data=1)

    s0_predicate = Predicate(lambda s: s == s0)
    s1_predicate = Predicate(lambda s: s == s1)
    policy = lambda s: 'a0' if s == s0 else 'a1' #shouldn't actually matter

    o0 = Option(s0_predicate, s1_predicate, policy, name='0 --> 1')
    o1 = Option(s1_predicate, s0_predicate, policy, name='1 --> 0')

    def reward_func(state, action):
        if state == s0 and action == o0:
            return 1
        if state == s1 and action == o1:
            return 1
        return 0

    def transition_func(state, action):
        if state == s0 and action == o0:
            return s1
        if state == s1 and action == o1:
            return s0
        return state

    omdp = OptionsMDP([o0, o1], transition_func, reward_func, s0)
    omdp_vi = OptionsMDPValueIteration(omdp, delta=0.005, max_iterations=1000, sample_rate=5)
    iters, val = omdp_vi.run_vi()
