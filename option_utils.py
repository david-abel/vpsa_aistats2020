from collections import defaultdict
import numpy as np
import random
import networkx as nx

# simple_rl imports.
from simple_rl.planning import ValueIteration
from simple_rl.abstraction.action_abs.InListPredicateClass import InListPredicate
from simple_rl.abstraction.action_abs.OptionClass import Option
from simple_rl.mdp import MDP, State
from simple_rl.abstraction.action_abs.PolicyFromDictClass import PolicyFromDict
from simple_rl.abstraction.action_abs.PredicateClass import Predicate
# Local imports.
from abstr_helpers import make_dict_from_lambda

def make_fixed_rand_options(mdp, state_abstr):
    '''
    Args:
        mdp (simple_rl.MDP)
        state_abstr (simple_rl.StateAbstraction)

    Returns:
        (list)
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
                state_pairs[(s_a, s_a_prime)] = 1    # Grab relevant states.
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

def rollout(option, cur_state, reward_func, transition_func, max_rollout_depth, gamma):
    '''
    Summary:
        Executes the option until termination.

    Args:
        option (simple_rl.abstraction.action_abs.Option)
        cur_state (simple_rl.State)
        reward_func (lambda)
        transition_func (lambda)
        max_rollout_depth (int)
        gamma (float)

    Returns:
        (tuple):
            1. (State): state we landed in.
            2. (float): Reward from the trajectory.
    '''
    total_reward = 0
    rollout_depth = 0
    discount = 1

    if option.is_init_true(cur_state):
        # Act until terminal.
        while not option.is_term_true(cur_state) and not cur_state.is_terminal() and rollout_depth < max_rollout_depth:
            next_state = transition_func(cur_state, option.act(cur_state))
            total_reward += discount * reward_func(cur_state, option.act(cur_state), next_state)
            cur_state = next_state
            rollout_depth += 1
            discount *= gamma

    return cur_state, total_reward, rollout_depth


def compute_option_models(mdp, vi, abstr_state, state_abstr, option, sample_rate=10):
    """
    Computes the reward model, termination probability distribution, multitime model, and Q^*_{s_\phi} for an option
    """
    rew_func = mdp.get_reward_func()
    trans_func = mdp.get_transition_func()

    # s : R_o(s, o)
    r_dict = {}

    # s : t : s' : p(s', t | s, o)
    p_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for state in state_abstr.get_lower_states_in_abs_state(abstr_state):
        rewards = []

        for i in range(sample_rate):
            s_prime, disc_reward, time = rollout(option, state, rew_func, trans_func, max_rollout_depth=20,
                                                 gamma=mdp.get_gamma())
            rewards.append(disc_reward)
            p_dict[state][time][s_prime] += 1

        r_dict[state] = float(sum(rewards)) / sample_rate

        # Normalize p table
        for t in p_dict[state]:
            for s_prime in p_dict[state][t]:
                p_dict[state][t][s_prime] = float(p_dict[state][t][s_prime]) / sample_rate

    # Option reward model from Sutton et al 1999
    def rew_lambda(s):
        return r_dict[s]

    # Probability that the option terminates in s_prime after t steps
    def term_prob_lambda(s, t, s_prime):
        return p_dict[s][t][s_prime]

    # s : s' : T_o(s' | s, o)
    mtm_dict = defaultdict(lambda: defaultdict(float))
    for state in p_dict:
        for t in p_dict[state]:
            for s_prime in p_dict[state][t]:
                mtm_dict[state][s_prime] += p_dict[state][t][s_prime] * (mdp.get_gamma() ** t)

    # Multi-time model of Sutton et al 1999
    def mtm_lambda(s, s_prime):
        return mtm_dict[s][s_prime]

    q_star_dict = {}
    for s in r_dict:
        # Compute Q^*(s, o) = R_o(s, o) + \sum_{s'} T_o(s' | s, o) V*(s')
        q_star = r_dict[s]
        for s_prime in mtm_dict[s]:
            q_star += mtm_dict[s][s_prime] * vi.get_value(s_prime)
        q_star_dict[s] = q_star

    return r_dict, p_dict, mtm_dict, q_star_dict


def policy_to_dict(states, policy):
    policy_dict = {}
    for state in states:
        policy_dict[state] = policy(state)
    return policy_dict


def find_eigenoptions(mdp, num_options=4, init_everywhere=False):
    delta = 0.001 # threshold for float point error
    
    # TODO: assume that the state-space is strongly connected.

    # Compute laplacian.
    A, state_to_id, id_to_state = get_transition_matrix(mdp)
    for n in range(A.shape[0]):
        if A[n][n] == 1:
            A[n][n] = 0 # Prune self-loops for the analysis            
    degrees = np.sum(A, axis=0)
    T = np.diag(degrees)
    Tngsqrt = np.diag(1.0 / np.sqrt(degrees))
    L = T - A
    normL = np.matmul(np.matmul(Tngsqrt, L), Tngsqrt)
    eigenvals, eigenvecs = np.linalg.eigh(normL)
    eigenoptions = []

    for i in range(0, num_options):
        # 1st eigenval is not useful
        maxnode = np.argwhere(eigenvecs[:,i] >= np.amax(eigenvecs[:, i]) - delta) + 1
        minnode = np.argwhere(eigenvecs[:,1] <= np.amin(eigenvecs[:, 1]) + delta) + 1

        # Make init/goal sets.
        init_set_nums = list(maxnode.flatten())
        init_set = [id_to_state[s - 1] for s in init_set_nums]
        goal_set_nums = list(minnode.flatten())
        goal_set = [id_to_state[s - 1] for s in goal_set_nums]

        # Define predicates.
        if init_everywhere:
            # Initiate everywhere.
            init_predicate = Predicate(lambda x:True)
        else:
            # Terminate everywhere
            init_predicate = InListPredicate(ls=init_set)
        term_predicate = InListPredicate(ls=goal_set)

        eigen_o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=make_option_policy(mdp, id_to_state.values(), goal_set))

        eigenoptions.append(eigen_o)


        # TODO: translate to an Option object.

    return eigenoptions[0:num_options]
    # TODO: normL * eigenvec = eigenval * eigenvec;


def find_betweenness_options(mdp, t=0.1, init_everywhere=False):
    T, state_to_id, id_to_state = get_transition_matrix(mdp)

    # print("T=", T)
    G = nx.from_numpy_matrix(T)
    N = G.number_of_nodes()
    M = G.number_of_edges()
    # print("nodes=", N)
    # print("edges=", M)

    #########################
    ## 1. Enumerate all candidate subgoals
    #########################
    subgoal_set = []
    for s in G.nodes():
        # print("s=", s)
        csv = nx.betweenness_centrality_subset(G, sources=[s], targets=G.nodes())
        # csv = nx.betweenness_centrality(G)
        # print("csv=", csv)
        for v in csv:
            if (s is not v) and (csv[v] / (N-2) > t) and (v not in subgoal_set):
                subgoal_set.append(v)

    # for s in subgoal_set:
    #     print(s, " is subgoal")
    # n_subgoals = sum(subgoal_set)
    # print(n_subgoals, "goals in total")
    # centralities = nx.betweenness_centrality(G)
    # for n in centralities:
    #     print("centrality=", centralities[n])

    #########################
    ## 2. Generate an initiation set for each subgoal
    #########################
    initiation_sets = defaultdict(list)
    support_scores = defaultdict(float)
    
    for g in subgoal_set:
        csg = nx.betweenness_centrality_subset(G, sources=G.nodes(), targets=[g])
        score = 0
        for s in G.nodes():
            if csg[s] / (N-2) > t:
                initiation_sets[g].append(s)
                score += csg[s]
        support_scores[g] = score
                
    # for g in subgoal_set:
    #     print("init set for ", g, " = ", initiation_sets[g])

    #########################
    ## 3. Filter subgoals according to their supports
    #########################
    filtered_subgoals = []

    subgoal_graph = G.subgraph(subgoal_set)
    
    sccs = nx.connected_components(subgoal_graph) # TODO: connected components are used instead of SCCs
    # sccs = nx.strongly_connected_components(G)
    for scc in sccs:
        scores = []
        goals = []
        for n in scc:
            scores.append(support_scores[n])
            goals.append(n)
            # print("score of ", n, " = ", support_scores[n])
        # scores = [support_scores[x] for x in scc]
        best_score = max(scores)
        best_goal = goals[scores.index(best_score)]
        filtered_subgoals.append(best_goal)

    options = []
    for g in filtered_subgoals:
        init_set_nums = initiation_sets[g]
        goal_set_nums = [g]
        init_set = [id_to_state[s] for s in init_set_nums]
        goal_set = [id_to_state[s] for s in goal_set_nums]


        # Define predicates.
        if init_everywhere:
            # Initiate everywhere.
            init_predicate = Predicate(lambda x:True)
        else:
            # Terminate everywhere
            init_predicate = InListPredicate(ls=init_set)
        term_predicate = InListPredicate(ls=goal_set)

        between_o = Option(init_predicate=init_predicate,
                       term_predicate=term_predicate,
                       policy=make_option_policy(mdp, id_to_state.values(), goal_set))

        options.append(between_o)

    return options

def make_option_policy(mdp, init_states, goal_states):
    '''
    Args:
        mdp
        init_states
        goal_states

    Returns:
        (lambda)
    '''

    def goal_new_trans_func(s, a):
        original = s.is_terminal()
        s.set_terminal(s in goal_states) # or original)
        s_prime = mdp.get_transition_func()(s, a)
        s_prime.set_terminal(s_prime in goal_states)
        s.set_terminal(original)
        return s_prime

    # Make a new MDP.
    mini_mdp = MDP(actions=mdp.get_actions(),
            init_state=mdp.get_init_state(),
            transition_func=goal_new_trans_func,
            reward_func=lambda x,y,z : -1)

    o_policy, _ = _make_mini_mdp_option_policy(mini_mdp)

    return o_policy


def _make_mini_mdp_option_policy(mini_mdp):
    '''
    Args:
        mini_mdp (MDP)

    Returns:
        Policy
    '''
    # Solve the MDP defined by the terminal abstract state.
    mini_mdp_vi = ValueIteration(mini_mdp, delta=0.005, max_iterations=500, sample_rate=20)
    iters, val = mini_mdp_vi.run_vi()

    o_policy_dict = make_dict_from_lambda(mini_mdp_vi.policy, mini_mdp_vi.get_states())
    o_policy = PolicyFromDict(o_policy_dict)

    return o_policy.get_action, mini_mdp_vi


def get_transition_matrix(mdp):
    '''
    Args:
        mdp

    Returns:
        T (list): transition matrix
        state_to_id (dict)
        id_to_state (dict)
    '''
    vi = ValueIteration(mdp)  # Use VI class to enumerate states
    vi.run_vi()
    vi._compute_matrix_from_trans_func()
    # q = vi.get_q_function()
    trans_matrix = vi.trans_dict

    state_to_id = {}
    id_to_state = {}
    for i, u in enumerate(trans_matrix):
        state_to_id[u] = i
        id_to_state[i] = u

    T = np.zeros((len(trans_matrix), len(trans_matrix)), dtype=np.int8)
    for i, u in enumerate(trans_matrix):
        for j, a in enumerate(trans_matrix[u]):
            for k, v in enumerate(trans_matrix[u][a]):
                if trans_matrix[u][a][v] > 0:
                    T[i][state_to_id[v]] = 1  # Node index starts from 1 (Minizinc is 1-indexed language)
    return T, state_to_id, id_to_state