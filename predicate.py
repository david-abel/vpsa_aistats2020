from simple_rl.planning import ValueIteration
from data_structs.OptionsMDPClass import OptionsMDP

N = 2

def hierarch_four_rooms_l2(state_a, state_b, mdp):
    width, height = mdp.width, mdp.height
    if (state_a.x <= width / 2) == (state_b.x <= width / 2) \
        and (state_a.y <= height / 2) == (state_b.y <= height / 2):
        return True
    return False

def hierarch_four_rooms_l1(state_a, state_b, mdp):
    '''
    Summary:
        Breaks a grid world into a 4x4 tiling.
    '''
    width, height = mdp.width + 1, mdp.height + 1

    if int(state_a.x * 4.0 / width) == int(state_b.x * 4.0 / width) \
        and int(state_a.y * 4.0 / height) == int(state_b.y * 4.0 / height):
        return True
    return False

def identity_predicate(state_a, state_b, mdp):
    return False

def bisimulation_predicate(state_a, state_b, mdp):
    if isinstance(mdp, OptionsMDP):
        actions = mdp.get_actions(state_a)
    else:
        actions = mdp.get_actions()

    for action in actions:
        if mdp.transition_func(state_a, action) != mdp.transition_func(state_b, action):
            return False
        if mdp.reward_func(state_a, action) != mdp.reward_func(state_b, action):
            return False

    return True

def four_rooms_predicate_9x9(state_a, state_b, mdp):
    def quadrant(state):
        x,y = state.data[0], state.data[1]
        if x <= 4 and y <= 5:
            # Bottom left.
            return 0
        elif x > 4 and y <= 3:
            # Bottom right
            return 1
        elif x <= 5 and y > 5:
            # Top left.
            return 2
        else:
            return 3
    if quadrant(state_a) == quadrant(state_b):
        return True
    return False

def four_rooms_predicate_11x11(state_a, state_b, mdp):
    def quadrant(state):
        x,y = state.data[0], state.data[1]
        if x <= 5 and y <= 6:
            # Bottom left.
            return 0
        elif x >= 6 and y < 5:
            # Bottom right.
            return 1
        elif x <= 6 and y > 6:
            # Top left.
            return 2
        else:
            return 3
    if quadrant(state_a) == quadrant(state_b):
        return True
    return False


def test_predicate(state_a, state_b, mdp):
    # Hacky test predicate to enable creation of a multilevel hierarchy.
    if isinstance(state_a.data, list) and isinstance(state_b.data, list):
        return state_a.data[0] == state_b.data[0]
    return int(state_a.data[-1:]) / 2 == int(state_b.data[-1:]) / 2


def same_rew_and_reachable_in_n_steps_predicate(state_a, state_b, mdp):
    if isinstance(mdp, OptionsMDP):
        actions = mdp.get_actions(state_a)
    else:
        actions = mdp.get_actions()

    samples = 10
    for a in actions:
        for i in range(samples):
            if mdp.reward_func(state_a, a) != mdp.reward_func(state_b, a):
                return False

    b_from_a = False
    reachable_states = set([])
    cur_state = state_a
    for i in range(N):
        for a in mdp.get_actions(cur_state):
            counter = 0
            next_state = mdp.transition_func(cur_state, a)
            while not a.is_term_true(next_state) and counter < 50:
                next_state = mdp.transition_func(cur_state, a)
                counter += 1

            if next_state == state_b:
                b_from_a = True

    if not b_from_a:
        return False

    cur_state = state_b
    for i in range(N):
        for a in mdp.get_actions(cur_state):
            counter = 0
            next_state = mdp.transition_func(cur_state, a)
            while not a.is_term_true(next_state) and counter < 50:
                next_state = mdp.transition_func(cur_state, a)
                counter += 1

            if next_state == state_a:
                return True

    return False


def reachable_in_n_steps_predicate(state_a, state_b, mdp):
    # TODO (transitive version? --> communnicating MDP)

    b_from_a = False
    reachable_states = set([])
    next_state = state_a
    for i in range(N):
        for a in mdp.get_actions():
            next_state = mdp.transition_func(next_state, a)

            if next_state == state_b:
                b_from_a = True

    if not b_from_a:
        return False

    next_state = state_b
    for i in range(N):
        for a in mdp.get_actions():
            next_state = mdp.transition_func(next_state, a)

            if next_state == state_b:
                return True

    return False
