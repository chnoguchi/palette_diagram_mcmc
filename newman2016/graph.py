
import numpy as np

class MyGraph:
	# N				# Number of nodes
	# l				# Pair of nodes(i<j)

	def getNumberOfLinks(self):
		return self._l.shape[0]

	def initMembers(self, N, links):
		self._N = N
		self._l = links

		



