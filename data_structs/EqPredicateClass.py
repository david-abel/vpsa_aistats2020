class EqPredicate(object):

	def __init__(self, y, func=None):
		self.y = y
		if func is None:
			self.func = lambda a : a
		else:
			self.func = func

	def is_true(self, x):
		return self.func(x) == self.y
