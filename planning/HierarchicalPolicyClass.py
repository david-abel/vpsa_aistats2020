# Python imports.
import random

# Local imports.
from planning.OptionsMDPValueIterationClass import OptionsMDPValueIteration


class HierarchicalPolicy(object):

    def __init__(self, mdp_hierarchy):
        self.mdp_hierarchy = mdp_hierarchy
        top_mdp = mdp_hierarchy.get_level_n_mdp(-1)
        self.op_vi = OptionsMDPValueIteration(top_mdp, delta=0.005, max_iterations=1000, sample_rate=30)
        self.op_vi.run_vi()

    def policy(self, ground_state):
        # Move up via phi.
        state_at_each_level = self.mdp_hierarchy.phi_zero_to_n(ground_state)

        # Get the top level action.
        next_option = random.choice(self.op_vi.get_max_q_actions(state_at_each_level[-1]))

        # Move down via action abstraction.
        for i, action_abstr in enumerate(reversed(self.mdp_hierarchy.aa_list)):
            next_option = next_option.act(state_at_each_level[-i - 2])

        return next_option.act(ground_state)
