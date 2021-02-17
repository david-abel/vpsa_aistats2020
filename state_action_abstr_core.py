# Python imports.
import random
import sys
import numpy as np
from collections import defaultdict

# Other imports.
import abstr_helpers as ah
from simple_rl.mdp import State
from simple_rl.abstraction.state_abs.StateAbstractionClass import StateAbstraction
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.ActionAbstractionClass import ActionAbstraction
from simple_rl.mdp.MDPClass import MDP
from simple_rl.mdp.StateClass import State
from simple_rl.mdp.MDPDistributionClass import MDPDistribution
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from data_structs.EqPredicateClass import EqPredicate
from data_structs.NeqPredicateClass import NeqPredicate
from data_structs.MDPHierarchyClass import MDPHierarchy
from data_structs.RewardFuncDictClass import RewardFuncDict, make_rew_dict_from_lambda
from data_structs.TransitionFuncDictClass import TransitionFuncDict, make_trans_dict_from_lambda
from data_structs.OptionsMDPClass import OptionsMDP


# ===================================
# == (1) Compute State Abstraction ==
# ===================================

def compute_phi_given_m(m, predicate, level, states):
    '''
    Args:
        m (simple_rl.MDP)

    Returns:
        phi (simple_rl.abstraction.StateAbstraction)
    '''
    # Group states according to given predicate
    phi = {}
    abstr_state_idx = 0
    for i in range(len(states)):
        in_existing_cluster = False
        for j in range(i):
            states_equiv = predicate(states[i], states[j], m)
            if states_equiv:
                phi[states[i]] = phi[states[j]]
                in_existing_cluster = True

        if not in_existing_cluster:
            phi[states[i]] = State(data='lvl' + str(level) + '_' + str(abstr_state_idx))
            abstr_state_idx += 1
    print "\t\t|S|", len(states)
    print "\t\t|S_phi|", abstr_state_idx


    return StateAbstraction(phi)


# ====================================
# == (2) Compute Action Abstraction ==
# ====================================

def compute_omega_given_m_phi(mdp, state_abstr):
    '''
    Args:
        mdp (simple_rl.MDP)
        phi (simple_rl.abstraction.StateAbstraction)

    Returns:
        omega (simple_rl.abstraction.ActionAbstraction)
    '''
    # Grab relevant states.
    abs_states = state_abstr.get_abs_states()
    g_start_state = mdp.get_init_state()

    # Compute all directed options that transition between abstract states.
    options = []
    state_pairs = {}
    placeholder_policy = lambda s : random.choice(mdp.get_actions(s))

    # For each s_{phi,1} s_{phi,2} pair.
    for s_a in abs_states:
        for s_a_prime in abs_states:
            if not(s_a == s_a_prime) and (s_a,s_a_prime) not in state_pairs.keys() and (s_a_prime, s_a) not in state_pairs.keys():
                # Make an option to transition between the two states.
                init_predicate = InListPredicate(ls=state_abstr.get_ground_states_in_abs_state(s_a))
                term_predicate = InListPredicate(ls=state_abstr.get_ground_states_in_abs_state(s_a_prime))
                
                o = Option(init_predicate=init_predicate,
                           term_predicate=term_predicate,
                           policy=placeholder_policy)

                options.append(o)
                state_pairs[(s_a, s_a_prime)] = 1

    # Prune.
    pruned_option_set = ah._prune_redundant_options(options, state_pairs.keys(), state_abstr, mdp)

    return ActionAbstraction(options=pruned_option_set, on_failure="primitives")


# ==============================
# == (3) Compute Abstract MDP ==
# ==============================

def compute_abstr_mdp_given_m_phi_omega(mdp, state_abstr, action_abstr, max_rollout_depth=50, sample_rate=25):
    '''
    Args:
        mdp (simple_rl.MDP)
        state_abstr (simple_rl.abstraction.StateAbstraction)
        action_abstr (simple_rl.abstraction.ActionAbstraction)
        max_rollout_depth (int)

    Returns:
        mdp (simple_rl.MDP)
    '''
    # Compute R and T lambdas.
    abstr_reward_lambda = ah.make_abstr_reward_lambda(mdp, state_abstr, action_abstr, max_rollout_depth, sample_rate)
    abstr_transition_lambda = ah.make_abstr_transition_lambda(mdp, state_abstr, action_abstr, max_rollout_depth, sample_rate)

    # Make the components of the Abstract MDP.
    abstr_init_state = state_abstr.phi(mdp.get_init_state())
    abstr_action_space = action_abstr.get_actions()
    abstr_state_space = state_abstr.get_abs_states()

    print "\tCompute abstr reward"
    abstr_reward_func = RewardFuncDict(abstr_reward_lambda, abstr_state_space, abstr_action_space, sample_rate=sample_rate)

    print "\tCompute abstr transition"
    abstr_transition_func = TransitionFuncDict(abstr_transition_lambda, abstr_state_space, abstr_action_space, sample_rate=sample_rate)

    # Convert option initiation and termination conditions to abstract space.
    for option in abstr_action_space:
        option.init_predicate = EqPredicate(option.init_predicate.y, lambda s: s)
        option.term_predicate = NeqPredicate(option.term_predicate.y, lambda s: s)

    # Make the MDP.
    abstr_mdp = OptionsMDP(actions=abstr_action_space,
                           init_state=abstr_init_state,
                           reward_func=abstr_reward_func.reward_func,
                           transition_func=abstr_transition_func.transition_func,
                           gamma=1.0)

    return abstr_mdp


# ===========================
# == (4) Compute Hierarchy ==
# ===========================

def make_hierarchy(mdp, phi_predicate, max_level=None):
    '''
    Args:
        mdp (simple_rl.MDP)
        phi_predicate (lambda : simple_rl.State x simple_rl.State --> bool)

    Returns:
        mdp_hierarch (MDPHierarchy)
    '''
    prev_mdp = mdp
    mdp_hierarch = MDPHierarchy(mdp_list=[prev_mdp], sa_list=[], aa_list=[])

    level = 0
    while True:
        print "\n" + "~"*20 + "\n~~ Making Level", str(level) + " ~~\n" + "~" * 20
        # Explore state space
        exploration_sample_rate = 50
        states, state_graph = ah.compute_reachable_state_space(prev_mdp, exploration_sample_rate)

        # Make State Abstraction.
        print "\n(lvl" + str(level) + ".1) Computing phi given m"
        phi = compute_phi_given_m(prev_mdp, phi_predicate, level, states)
        ah.visualize_graph_phi(state_graph, phi._phi)

        # Show state space sizes.
        print "\t |S|_phi_" + str(level) + ":", phi.get_num_ground_states()
        print "\t |S|_phi_" + str(level + 1) + ":", phi.get_num_abstr_states()

        # Visualize.
        if level == 0:
            print "\n" + "-"*10
            ah.visualize_gridworld_phi(prev_mdp, phi._phi)
            print "-"*10 + "\n"

        # Convergence check.
        if is_hierarchy_converged(phi):
            print "\n" + "=" * 30
            print "== Hierarchy converged with", level, "levels =="
            print "=" * 30 + "\n"
            break

        # Make Action Abstraction.
        print "\n(lvl" + str(level) + ".2) Computing omega given m, phi"
        omega = compute_omega_given_m_phi(prev_mdp, phi)

        # Make Abstract MDP.
        print "\n(lvl" + str(level) + ".3)Computing Abstract MDP"
        next_mdp = compute_abstr_mdp_given_m_phi_omega(prev_mdp, phi, omega, max_rollout_depth=20)

        # Update hierarchy.
        mdp_hierarch.add_level(state_abstr=phi, action_abstr=omega, next_mdp=next_mdp)

        for s_a in phi.get_abs_states():
            if len(next_mdp.get_actions(s_a)) == 0:
                ground_states = mdp_hierarch.get_all_level_0_states_in_abs_state(s_a)
                print "Zero options in abs state:", s_a, "ground:", ground_states

        level += 1
        prev_mdp = next_mdp

    return mdp_hierarch


# ================================
# == (5) Is Hierarchy Converged ==
# ================================

def is_hierarchy_converged(phi):
    '''
    Args:
        prev_mdp (simple_rl.MDP)
        next_mdp (simple_rl.MDP)

    Returns:
        (bool)
    '''

    # Check if we can compress the state space further.
    return phi.get_num_abstr_states() == 1 or phi.get_num_ground_states() == phi.get_num_abstr_states()
