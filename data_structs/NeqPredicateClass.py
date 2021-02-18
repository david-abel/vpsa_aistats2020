class NeqPredicate(object):

	def __init__(self, y, func):
		self.y = y
		self.func = func

	def is_true(self, x):
		return not(self.func(x) == self.y)
