from simple_rl.mdp import MDP
try:
   import cPickle as pickle
except:
   import pickle

PICKLE_FILE_NAME = "mdp_hierarch.pickle"

class MDPHierarchy(MDP):

	def __init__(self, sa_list, aa_list, mdp_list):
		'''
		Args:
			sa_list (list)
			aa_list (list)
			mdp_list (list)
		'''
		self.sa_list = sa_list
		self.aa_list = aa_list
		self.mdp_list = mdp_list

	# ---------------
	# -- Accessors --
	# ---------------

	def get_level_n_sa(self, n):
		return self.sa_list[n]

	def get_level_n_aa(self, n):
		return self.aa_list[n]

	def get_level_n_mdp(self, n):
		return self.mdp_list[n]

	def get_all_level_0_states_in_abs_state(self, abs_state):
		level = int(str(abs_state)[str(abs_state).index("lvl") + 3:str(abs_state).index("_")])

		cur_abstr_states = [abs_state]
		while level > -1:
			next_states = []
			for s_a in cur_abstr_states:
				next_states += self.sa_list[level].get_ground_states_in_abs_state(s_a)
			cur_abstr_states = next_states[:]
			level -= 1

		return cur_abstr_states

	def get_top_level_state_space(self):
		return self.sa_list[-1].get_abs_states()

	def add_level(self, state_abstr, action_abstr, next_mdp):
		self.sa_list.append(state_abstr)
		self.aa_list.append(action_abstr)
		self.mdp_list.append(next_mdp)


	def phi_zero_to_n(self, ground_state):
		'''
		Args:
			ground_state (simple_rl.State)

		Returns:
			(list of simple_rl.State)
		'''
		states = [ground_state]
		state = ground_state
		for state_abstr in self.sa_list:
			state = state_abstr.phi(state)
			states.append(state)

		return states

def save_hierarchy(mdp_hierarchy):
	hierarch_file = open(PICKLE_FILE_NAME,'wb')
	pickle.dump(mdp_hierarchy, hierarch_file)
	hierarch_file.close()

def load_hierarchy(self):
	hierarch_file = open(PICKLE_FILE_NAME,'wb')
	mdp_hierarchy = pickle.load(PICKLE_FILE_NAME)
	hierarch_file.close()

	return mdp_hierarchy
