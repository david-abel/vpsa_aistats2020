"""
phi_relative_options_experiments.py

Code associated with the 2020 AISTATS paper:
    Value Preserving State-Action Abstractions
    David Abel, Nathan Umbanhowar, Khimya Khetarpal
    Dilip Arumugam, Doina Precup, Michael L. Littman
"""

# Python imports.
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
plt.rcParams['font.family'] = 'sans-serif'
PLOT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
import numpy as np

# simple_rl imports.
from simple_rl.tasks import FourRoomMDP
from simple_rl.agents import QLearningAgent, RMaxAgent, DelayedQAgent
from simple_rl.agents import DoubleQAgent, FixedPolicyAgent, SarsaAgent
from simple_rl.abstraction import AbstractionWrapper
from simple_rl.abstraction.action_abs import ActionAbstraction
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.planning import ValueIteration
from simple_rl.run_experiments import run_agents_on_mdp, run_single_agent_on_mdp

# Local imports.
import state_action_abstr_core as core
import abstr_helpers as ah
from predicate import four_rooms_predicate_9x9, four_rooms_predicate_11x11, reachable_in_n_steps_predicate
from data_structs.EqPredicateClass import EqPredicate
from data_structs.NeqPredicateClass import NeqPredicate
from option_utils import find_eigenoptions


###############################################
################# POLICIES ####################
###############################################

def get_fixed_random_policy(mdp):
    policy_map = {}
    def fixed_random_policy(state):
        '''
        Random policy that gives the same action for the same s whenever it is called.
        '''
        if state not in policy_map:
            act = random.choice(mdp.get_actions())
            policy_map[state] = act
            return act
        return policy_map[state]
    return fixed_random_policy


def get_eps_greedy_policy(eps, policy, actions):
    def eps_greedy_policy(state):
        if random.random() < eps:
            return np.random.choice(actions)
        return policy(state)
    return eps_greedy_policy


#####################################################################
################# PHI-RELATIVE OPTIONS FUNCTIONS ####################
#####################################################################


def make_phi_relative_options(mdp, state_abstr, options_per_s_phi=5):
    '''
    Args:
        mdp (simple_rl.MDP)
        state_abstr (simple_rl.StateAbstraction)
        option_epsilon (float)
        options_per_s_phi (int)

    Returns:
        (list)
    '''

    options = []

    # For each abstract state.
    for s_phi in state_abstr.get_abs_states():

        for option in range(options_per_s_phi):
            # Make an option to transition between the two states.
            init_predicate = EqPredicate(y=s_phi, func=state_abstr.phi)
            term_predicate = NeqPredicate(y=s_phi, func=state_abstr.phi)
            next_option = Option(init_predicate=init_predicate,
                                 term_predicate=term_predicate,
                                 policy=get_fixed_random_policy(mdp))
            options.append(next_option)

    return options


def make_single_action_phi_relative_options(mdp, state_abstr):
    """
    For every s_phi, constructs a phi-relative option corresponding to
    each action that takes that action everywhere within s_phi.
    """
    options = []
    for s_phi in state_abstr.get_abs_states():
        actions = mdp.get_actions()
        for action in actions:
            init_predicate = EqPredicate(y=s_phi, func=state_abstr.phi)
            term_predicate = NeqPredicate(y=s_phi, func=state_abstr.phi)
            # See https://stackoverflow.com/questions/19837486/python-lambda-in-a-loop
            # for why the lambda is constructed like this.
            o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=lambda s, bound_action=action: bound_action)
            options.append(o)
    return options

def make_fixed_random_options(mdp, state_abstr, num_options_per_s_a=2):
    """
    Args:
        mdp
        state_abstr

    Returns:
        (list)
    """

    options = []
    for s_phi in state_abstr.get_abs_states():
        init_predicate = EqPredicate(y=s_phi, func=state_abstr.phi)
        term_predicate = NeqPredicate(y=s_phi, func=state_abstr.phi)

        for _ in range(num_options_per_s_a):
            o_rand = Option(init_predicate=init_predicate,
                            term_predicate=term_predicate,
                            policy=get_fixed_random_policy(mdp))
            options.append(o_rand)

    return options

def make_near_optimal_phi_relative_options(mdp, state_abstr, method='optimal', num_rand_opts=0, **kwargs):
    """
    Args:
        mdp
        state_abstr
        method
        num_rand_opts

    Returns:
        (list)
    """
    # Get the optimal Q function
    from planning.OptionsMDPValueIterationClass import OptionsMDPValueIteration
    from data_structs.OptionsMDPClass import OptionsMDP

    if isinstance(mdp, OptionsMDP):
        value_iter = OptionsMDPValueIteration(mdp, sample_rate=20)
    else:
        value_iter = ValueIteration(mdp, sample_rate=10)

    value_iter.run_vi()

    options = []
    optimal_options = []
    for s_phi in state_abstr.get_abs_states():
        init_predicate = EqPredicate(y=s_phi, func=state_abstr.phi)
        term_predicate = NeqPredicate(y=s_phi, func=state_abstr.phi)
        o_star = Option(init_predicate=init_predicate,
                        term_predicate=term_predicate,
                        policy=lambda s: value_iter.policy(s))

        if method == 'optimal':
            options.append(o_star)
        if method == 'eps-greedy':
            eps = kwargs['eps']

            eps_greedy_policy = get_eps_greedy_policy(eps, value_iter.policy, mdp.get_actions())

            o_eps = Option(init_predicate=init_predicate,
                           term_predicate=term_predicate,
                           policy=eps_greedy_policy)


            for _ in range(num_rand_opts):
                o_rand = Option(init_predicate=init_predicate,
                                term_predicate=term_predicate,
                                policy=lambda x: random.choice(mdp.get_actions()))
                options.append(o_rand)

            options.append(o_eps)
        else:
            options.append(o_star)

    return options, optimal_options


############################################
################# OTHER ####################
############################################

def evaluate_policy(mdp, state, policy, rollouts=100, max_trajectory_length=200):
    """ Monte Carlo estimate of V^{pi}(state) """
    reward_func = mdp.get_reward_func()
    trans_func = mdp.get_transition_func()

    total_return = 0

    for _ in range(rollouts):
        step = 0
        cur_state = state
        discount = 1
        ret = 0

        while step < max_trajectory_length and not cur_state.is_terminal():
            action = policy(cur_state)
            next_state = trans_func(cur_state, action)
            reward = reward_func(cur_state, action, next_state)
            ret += discount * reward

            discount *= mdp.get_gamma()
            cur_state = next_state
            step += 1

        total_return += ret

    return float(total_return) / rollouts


##################################################
################# EXPERIMENTS ####################
##################################################


def run_learning_experiment():
    """
    Summary:
        Builds different sets of options and contrasts how RL algorithms
        perform when learning with them.
    """
    # Define MDP.
    width, height = 11, 11
    mdp = FourRoomMDP(width=width, height=height, goal_locs=[(width, height)], slip_prob=0.05)
    actions = mdp.get_actions()

    # Make State Abstraction.
    states, _ = ah.compute_reachable_state_space(mdp, sample_rate=50)
    if isinstance(mdp, FourRoomMDP):
        predicate = four_rooms_predicate_11x11
    else:
        predicate = reachable_in_n_steps_predicate

    state_abstr = core.compute_phi_given_m(mdp, predicate, level=1, states=states)

    # Make initial Options.
    num_rand_opts_to_add = 2
    options, _ = make_near_optimal_phi_relative_options(mdp, state_abstr, 'eps-greedy', num_rand_opts=num_rand_opts_to_add, eps=0.05)
    action_abstr = ActionAbstraction(options=options, prim_actions=actions)
    action_abstr_w_prims = ActionAbstraction(options=options, prim_actions=actions, incl_primitives=True)

    # Find eigen options.
    # num_eigen_options = max(1, num_rand_opts_to_add - 1)
    # eigen_options_init_all = find_eigenoptions(mdp, num_options=num_eigen_options, init_everywhere=True)
    # eigen_options_w_prims = find_eigenoptions(mdp, num_options=num_eigen_options, init_everywhere=False)
    # eigen_aa_init_all = ActionAbstraction(options=eigen_options_init_all, prim_actions=actions, incl_primitives=False)
    # eigen_aa_w_prims = ActionAbstraction(options=eigen_options_w_prims, prim_actions=actions, incl_primitives=True)

    # Make agent.
    AgentClass = QLearningAgent #QLearningAgent #DoubleQAgent #DelayedQAgent
    ql_agent = AgentClass(mdp.get_actions())
    sa_aa_agent = AbstractionWrapper(AgentClass, agent_params={"actions": actions}, state_abstr=state_abstr, action_abstr=action_abstr_w_prims, name_ext="-$\\phi,O$")
    aa_agent = AbstractionWrapper(AgentClass, agent_params={"actions": actions}, state_abstr=None, action_abstr=action_abstr_w_prims, name_ext="-$O$")
    # aa_agent = AbstractionWrapper(AgentClass, agent_params={"actions": actions}, state_abstr=None, action_abstr=action_abstr_w_prims, name_ext="-$\\phi$")
    # Eigen agents.
    # eigen_agent_init_all = AbstractionWrapper(AgentClass, agent_params={"actions": actions}, state_abstr=None, action_abstr=eigen_aa_init_all, name_ext="-eigen_all")
    # eigen_agent_w_prims = AbstractionWrapper(AgentClass, agent_params={"actions": actions}, state_abstr=None, action_abstr=eigen_aa_w_prims, name_ext="-eigen_w_prims")
    agents = [ql_agent, aa_agent, sa_aa_agent] #, eigen_agent_init_all, eigen_agent_w_prims]

    # Run.
    if isinstance(mdp, FourRoomMDP):
        run_agents_on_mdp(agents, mdp, instances=10, episodes=500, steps=50)
    else:
        run_agents_on_mdp(agents, mdp, instances=10, episodes=100, steps=10)


def branching_factor_experiment(min_options=0, max_options=20, increment=2, instances=5, epsilon=0.05):
    '''
    Args:
        min_options (int)
        max_options (int)
        increment (int)

    Summary:
        Runs an experiment contrasting learning performance for different # options.
    '''
    # Define MDP.
    grid_size = 7
    mdp = FourRoomMDP(width=grid_size, height=grid_size, goal_locs=[(grid_size, grid_size)])

    # Make State Abstraction.
    states, _ = ah.compute_reachable_state_space(mdp, sample_rate=50)
    state_abstr = core.compute_phi_given_m(mdp, four_rooms_predicate_9x9, level=1, states=states)

    x_axis = range(min_options, max_options + 1, increment)
    y_axis = defaultdict(list) #[] #[0] * len(x_axis)
    conf_intervals = defaultdict(list)
    num_options_performance = defaultdict(lambda: defaultdict(list))

    # Choose dependent variable (either #steps per episode or #episodes).
    d_var_range = [(20, 5), (40, 250), (400, 2500)]

    for steps, episodes in d_var_range:
        print "steps, episodes", steps, episodes

        # Evaluate.
        for i, instance in enumerate(range(instances)):
            print "\tInstance", instance + 1, "of", str(instances) + "."

            # Make initial Options.
            for num_options in x_axis:

                options, _ = make_near_optimal_phi_relative_options(mdp, state_abstr, 'eps-greedy', num_rand_opts=num_options - 1, eps=epsilon)
                action_abstr = ActionAbstraction(options=options, prim_actions=mdp.get_actions())

                # Make agent.
                AgentClass = RMaxAgent # DoubleQAgent, QLearningAgent, SarsaAgent
                sa_aa_agent = AbstractionWrapper(AgentClass, agent_params={"actions": mdp.get_actions()}, state_abstr=state_abstr, action_abstr=action_abstr, name_ext="-$\\phi,O$")

                _, _, value_per_episode = run_single_agent_on_mdp(sa_aa_agent, mdp, episodes=episodes, steps=steps)
                mdp.reset()

                num_options_performance[(steps, episodes)][num_options].append(value_per_episode[-1])

    ############
    # Other types

    # Just state abstraction.
    steps, episodes = d_var_range[-1][0], d_var_range[-1][1]
    sa_agent = AbstractionWrapper(AgentClass, agent_params={"actions": mdp.get_actions()}, state_abstr=state_abstr, action_abstr=None, name_ext="-$\\phi$")
    _, _, value_per_episode = run_single_agent_on_mdp(sa_agent, mdp, episodes=episodes, steps=steps)
    num_options_performance[(d_var_range[-1][0], d_var_range[-1][1])]["phi"].append(value_per_episode[-1])   
    y_axis["phi"] = [value_per_episode[-1]]

    # Run random options.
    options = make_fixed_random_options(mdp, state_abstr)
    action_abstr = ActionAbstraction(options=options, prim_actions=mdp.get_actions())
    AgentClass = QLearningAgent
    rand_opt_agent = AbstractionWrapper(AgentClass, agent_params={"actions": mdp.get_actions()}, state_abstr=state_abstr, action_abstr=action_abstr, name_ext="-$\\phi,O_{\text{random}}$")
    _, _, value_per_episode = run_single_agent_on_mdp(rand_opt_agent, mdp, episodes=episodes, steps=steps)
    num_options_performance[(d_var_range[-1][0], d_var_range[-1][1])]["random"].append(value_per_episode[-1])   
    y_axis["random"] = [value_per_episode[-1]]

    # Makeoptimal agent.
    value_iter = ValueIteration(mdp)
    value_iter.run_vi()
    optimal_agent = FixedPolicyAgent(value_iter.policy)
    _, _, value_per_episode = run_single_agent_on_mdp(optimal_agent, mdp, episodes=episodes, steps=steps)
    y_axis["optimal"] = [value_per_episode[-1]]
    total_steps = d_var_range[0][0] * d_var_range[0][1]

    # Confidence intervals.
    for dependent_var in d_var_range:
        for num_options in x_axis:
            # Compute mean and standard error.
            avg_for_n = float(sum(num_options_performance[dependent_var][num_options])) / instances
            std_deviation = np.std(num_options_performance[dependent_var][num_options])
            std_error = 1.96 * (std_deviation / math.sqrt(len(num_options_performance[dependent_var][num_options])))
            y_axis[dependent_var].append(avg_for_n)
            conf_intervals[dependent_var].append(std_error)

    plt.xlabel("$|O_\\phi|$")
    plt.xlim([1, len(x_axis)])
    plt.ylabel("$V^{\hat{\pi}_{O_\\phi}}(s_0)$")
    plt.tight_layout() # Keeps the spacing nice.

    # Add just state abstraction.
    ep_val_del_q_phi = y_axis["phi"]
    label = "$O_{\\phi}$" #" N=1e" + str(str(total_steps).count("0")) + "$"
    plt.plot(x_axis, [ep_val_del_q_phi] * len(x_axis), marker="+", linestyle="--", linewidth=1.0, color=PLOT_COLORS[-1], label=label)

    # Add random options.
    ep_val_del_q = y_axis["random"]
    label = "$O_{random}$" #" N=1e" + str(str(total_steps).count("0")) + "$"
    plt.plot(x_axis, [ep_val_del_q] * len(x_axis), marker="x", linestyle="--", linewidth=1.0, color=PLOT_COLORS[0]) #, label=label)

    # Add optimal.
    ep_val_optimal = y_axis["optimal"]
    plt.plot(x_axis, [ep_val_optimal] * len(x_axis), linestyle="-", linewidth=1.0, color=PLOT_COLORS[1]) #, label="$\\pi^*$")


    for i, dependent_var in enumerate(d_var_range):
        total_steps = dependent_var[0] * dependent_var[1]
        label = "$O_{\\phi,Q_\\varepsilon^*}, N=1e" + str(str(total_steps).count("0")) + "$"
        plt.plot(x_axis, y_axis[dependent_var], marker="x", color=PLOT_COLORS[i + 2], linewidth=1.5, label=label)
        
        # Confidence intervals.
        top = np.add(y_axis[dependent_var], conf_intervals[dependent_var])
        bot = np.subtract(y_axis[dependent_var], conf_intervals[dependent_var])
        plt.fill_between(x_axis, top, bot, alpha=0.25, color=PLOT_COLORS[i + 2])


    plt.legend()
    plt.savefig("branching_factor_results.pdf", format="pdf")
    plt.cla()
    plt.close()


#########################################################################

def main():
    # branching_factor_experiment(1, 5, 1, instances=25, epsilon=0.05)
    run_learning_experiment()

if __name__ == "__main__":
    main()
