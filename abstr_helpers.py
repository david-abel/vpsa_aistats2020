# -*- coding: utf-8 -*-
# Python Imports
import random
from collections import defaultdict

try:
    import Queue
    queue = Queue
except NameError:
    import queue

# Other imports
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

# simple_rl imports.s
from simple_rl.mdp.MDPClass import MDP
from simple_rl.planning.ValueIterationClass import ValueIteration
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict

# Local impotrs.
from planning.OptionsMDPValueIterationClass import OptionsMDPValueIteration
from data_structs.OptionsMDPClass import OptionsMDP


# =================
# == R/T Lambdas ==
# =================

def make_abstr_reward_lambda(mdp, state_abstr, action_abstr, max_rollout_depth, sample_rate=10, step_cost=0.0):
    '''
    Args:
        mdp (simple_rl.MDP)
        state_abstr (simple_rl.abstraction.StateAbstraction)
        action_abstr (simple_rl.abstraction.ActionAbstraction)
        max_rollout_depth (int)

    Returns:
        (lambda)
    '''

    # Make abstract reward and transition functions.
    def abstr_reward_lambda(abstr_state, abstr_action, next_abstr_state=None):
        if abstr_state.is_terminal():
            return 0

        # Get relevant MDP components from the lower MDP.
        lower_states = state_abstr.get_lower_states_in_abs_state(abstr_state)

        lower_reward_func = mdp.get_reward_func()
        lower_trans_func = mdp.get_transition_func()

        # Compute reward.
        total_reward = 0
        for ground_s in lower_states:
            for sample in range(sample_rate):
                s_prime, reward = abstr_action.rollout(ground_s, lower_reward_func, lower_trans_func, max_rollout_depth=5, step_cost=step_cost)
                total_reward += float(reward) / (len(lower_states) * sample_rate) # Add weighted reward.

        return total_reward

    return abstr_reward_lambda


def make_abstr_transition_lambda(mdp, state_abstr, action_abstr, max_rollout_depth, sample_rate=10):
    '''
    Args:
        mdp (simple_rl.MDP)
        state_abstr (simple_rl.abstraction.StateAbstraction)
        action_abstr (simple_rl.abstraction.ActionAbstraction)
        max_rollout_depth (int)

    Returns:
        (lambda)
    '''

    def abstr_transition_lambda(abstr_state, abstr_action):
        is_ground_terminal = False
        for s_g in state_abstr.get_lower_states_in_abs_state(abstr_state):
            if s_g.is_terminal():
                is_ground_terminal = True
                # TODO: ?
                break

        # Get relevant MDP components from the lower MDP.
        if abstr_state.is_terminal():
            return abstr_state

        lower_states = state_abstr.get_lower_states_in_abs_state(abstr_state)
        lower_reward_func = mdp.get_reward_func()
        lower_trans_func = mdp.get_transition_func()

        # Compute next state distribution.
        s_prime_prob_dict = defaultdict(int)
        total_reward = 0
        for ground_s in lower_states:
            for sample in range(sample_rate):
                s_prime, reward = abstr_action.rollout(ground_s, lower_reward_func, lower_trans_func, max_rollout_depth)
                s_prime_prob_dict[s_prime] += (1.0 / (len(lower_states) * sample_rate)) # Weighted average.

        # Form distribution and sample s_prime.
        next_state_sample_list = list(np.random.multinomial(1, list(s_prime_prob_dict.values())).tolist())
        end_ground_state = list(s_prime_prob_dict.keys())[next_state_sample_list.index(1)]
        end_abstr_state = state_abstr.phi(end_ground_state)

        return end_abstr_state

    return abstr_transition_lambda


# ====================
# == Option Helpers ==
# ====================

def _prune_redundant_options(options, abstr_state_pairs, state_abstr, mdp):
    '''
    Args:
        Options(list)
        abstr_state_pairs (list)
        state_abstr (StateAbstraction)
        mdp (simple_rl.MDP)

    Returns:
        (list of Options)

    Summary:
        Removes redundant options. That is, if o_1 goes from s_A1 to s_A2, and
        o_2 goes from s_A1 *through s_A2 to s_A3, then we get rid of o_2.
    '''
    good_options = set([])

    # Remove the bad options.
    for i, o in enumerate(options):
        # print "\t  Option", i + 1, "of", len(options)

        # Make mini MDP.
        pre_abs_state, post_abs_state = abstr_state_pairs[i]
        ground_init_states = state_abstr.get_lower_states_in_abs_state(pre_abs_state)
        ground_reward_states = state_abstr.get_lower_states_in_abs_state(post_abs_state)
        mini_mdp = _make_mini_mdp(pre_abs_state, post_abs_state, state_abstr, mdp)

        # Make policy.
        o_policy, mini_mdp_vi = _make_mini_mdp_option_policy(mini_mdp, ground_init_states)
        opt_name = str(ground_init_states[0]) + "-" + str(ground_reward_states[0])
        o.set_name(opt_name)

        if _is_direct_option(o_policy, state_abstr, pre_abs_state, post_abs_state, mini_mdp_vi):
            # If it's a direct option, add it.
            o.set_policy(o_policy)
            good_options.add(o)


    print "\t Found", len(good_options), "good options."
    return good_options


def _is_direct_option(o_policy, state_abstr, pre_abs_state, post_abs_state, mini_mdp_vi):
    '''
    Args:
        o_policy
        state_abstr
        pre_abs_state
        post_abs_state
        mini_mdp_vi (simple_rl.ValueIteration)

    Returns:
        (bool)
    '''

    ground_init_states = state_abstr.get_lower_states_in_abs_state(pre_abs_state)

    # Compute overlap w.r.t. plans from each state.
    for init_g_state in ground_init_states:

        # TODO:
        # This was added because init_g_state was the goal state, which meant that mini_mdp_vi.plan stayed in
        # one state. Therefore, no option was being created from the abstract state containing the goal
        # state to any surrounding states. This broke compute_omega_given_m_phi at the next level of the hierarchy.
        # Is there a more elegant way to do this?
        if init_g_state.is_terminal():
            return True
        original = init_g_state.is_terminal()
        init_g_state.set_terminal(False)
        seq_reached_terminal = False

        # Prune overlapping ones.
        action_seq, state_seq = mini_mdp_vi.plan(state=init_g_state)
        for s_g in state_seq:

            if state_abstr.phi(s_g) not in [pre_abs_state, post_abs_state]:
                # Not a direct option.
                init_g_state.set_terminal(original)
                return False

            if state_abstr.phi(s_g) == post_abs_state:
                # We found a route to leave pre_abs and go to post_abs.
                seq_reached_terminal = True
                break

        if not seq_reached_terminal:
            # There was no route to go from pre_abs_state to post_abs_state.
            # We stayed in pre_abs_state the whole time.
            init_g_state.set_terminal(original)
            return False

    init_g_state.set_terminal(original)
    return seq_reached_terminal

def _make_mini_mdp(pre_abs_state, post_abs_state, state_abstr, mdp):
    '''
    Args:
        pre_abs_state (simple_rl.State)
        post_abs_state (simple_rl.State)
        state_abstr
        mdp (simple_rl.MDP)

    Returns:
        (simple_rl.MDP)
    '''

    # Get init and terminal lower level states.
    ground_init_states = state_abstr.get_lower_states_in_abs_state(pre_abs_state)
    ground_reward_states = state_abstr.get_lower_states_in_abs_state(post_abs_state)
    rand_init_g_state = random.choice(ground_init_states)

    # R and T for Option Mini MDP.
    def _directed_option_reward_lambda(s, a, s_prime):
        # TODO: might need to sample here?
        original = s.is_terminal()
        s.set_terminal(s not in ground_init_states)
        s_prime = mdp.transition_func(s, a)
        s.set_terminal(original)

        # Returns non-zero reward iff the action transitions to a new abstract state.
        return int(s_prime in ground_reward_states and not s in ground_reward_states)

    def new_trans_func(s, a):
        original = s.is_terminal()
        s.set_terminal(s not in ground_init_states)
        s_prime = mdp.transition_func(s, a)
        s.set_terminal(original)
        return s_prime

    mini_mdp = MDP(actions=mdp.get_actions(),
                          init_state=rand_init_g_state,
                          transition_func=new_trans_func,
                          reward_func=_directed_option_reward_lambda)

    return mini_mdp


def _make_mini_mdp_option_policy(mini_mdp, initiating_states):
    '''
    Args:
        mini_mdp (MDP)

    Returns:
        Policy
    '''
    # Solve the MDP defined by the terminal abstract state.
    if isinstance(mini_mdp, OptionsMDP):
        mini_mdp_vi = OptionsMDPValueIteration(mini_mdp, delta=0.005, max_iterations=1000, sample_rate=30)
    else:
        mini_mdp_vi = ValueIteration(mini_mdp, delta=0.005, max_iterations=1000, sample_rate=30)

    for s_g in initiating_states:
        if s_g.is_terminal():
            return lambda s: random.choice(mini_mdp.get_actions(s_g)), mini_mdp_vi

    iters, val = mini_mdp_vi.run_vi()
    o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, mini_mdp_vi.get_states() + initiating_states)
    o_policy = PolicyFromDict(o_policy_dict)

    return o_policy.get_action, mini_mdp_vi


def make_dict_from_lambda(policy_func, state_list):
    '''
    Args:
        policy_func (lambda : State --> str)
        state_list (list)

    Returns:
        (dict)
    '''
    if len(state_list) == 0: import pdb; pdb.set_trace()
    policy_dict = {}
    for s in state_list:
        policy_dict[s] = policy_func(s)

    return policy_dict


# ===============================
# == State Abstraction Helpers ==
# ===============================

def compute_reachable_state_space(mdp, sample_rate):
    '''
    Args:
        mdp (simple_rl.MDP)
        sample_rate (int)
    Returns:
        (set(simple_rl.State)): A set of all reachable states in the MDP.
    '''
    states = set()
    state_graph = nx.DiGraph()

    state_queue = queue.Queue()
    state_queue.put(mdp.init_state)
    states.add(mdp.init_state)

    while not state_queue.empty():
        s = state_queue.get()
        for a in mdp.get_actions():
            for samples in range(sample_rate): # Take @sample_rate samples to estimate E[V]
                next_state = mdp.transition_func(s, a)
                try:
                    state_graph.add_edge(s, next_state)
                    if next_state not in states:
                        states.add(next_state)
                        state_queue.put(next_state)
                except ValueError:
                    continue

    return list(states), state_graph

# ===========================
# == Visualization Helpers ==
# ===========================

def visualize_gridworld_phi(mdp, phi):
    '''
    Args:
        mdp (simple_rl.GridWorldMDP)
        phi (simple_rl.StateAbstraction)
    '''
    phi = phi._phi
    viz = np.full((mdp.width, mdp.height), u'█', dtype=object)
    for state, abstr_state in phi.items():
        coords = state.data

        val = int(abstr_state.data.split("_")[-1])
        viz[int(coords[0])-1, int(coords[1])-1] = chr(val + 48)
    viz = np.rot90(viz)
    for row in viz:
        print ' '.join(row)

def visualize_gridworld_phi_relative_options(mdp, phi, options):
    phi = phi._phi
    viz = np.full((mdp.height, mdp.width), u'█', dtype=object)
    for abstr_state in phi.values():
        # Find one representative ground state
        rep = None
        for s in phi.keys():
            if phi[s] == abstr_state:
                rep = s
                break

        # Find the options that can be initiated in this abstract state
        valid = [op for op in options if op.is_init_true(rep)]

        # For now, just pick a random one
        opt = np.random.choice(valid)

        # Create a map of actions to viz symbols
        symbol_map = {'left': u'←', 'right': u'→', 'up':u'↑', 'down':u'↓'}

        # Get ground states in abstract state and show opt's policy in the viz
        for state in [s for s in phi.keys() if phi[s] == abstr_state]:
            coords = state.data
            action = opt.policy(state)#.policy(state)
            viz[int(coords[0])-1, int(coords[1])-1] = symbol_map[action]

    viz = np.rot90(viz)
    for row in viz:
        print ' '.join(row)


def visualize_state_space(graph):
    pos = nx.kamada_kawai_layout(graph)
    nx.draw_networkx(graph, pos, arrows=True)
    plt.show()
    plt.gcf().clear()


def visualize_graph_phi(graph, phi):
    # Assign integer values to each abstract state for coloring
    abstr_states = list(set(phi.values()))
    abs_state_colors = {}
    for i, abstr_state in enumerate(abstr_states):
        abs_state_colors[abstr_state] = i

    # Build a list corresponding each ground state node to its abstract state color
    colors = []
    for state in graph.nodes():
        colors.append(abs_state_colors[phi[state]])

    # Make graph
    pos = nx.kamada_kawai_layout(graph)
    cmap = matplotlib.cm.get_cmap('rainbow')
    nx.draw_networkx(graph, pos, cmap=cmap, node_color=colors, vmin=0, vmax=len(abstr_states)-1, arrows=True)
    plt.show()
    plt.gcf().clear()
