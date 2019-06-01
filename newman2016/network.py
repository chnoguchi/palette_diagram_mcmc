
import numpy as np

class Network:

	# G		# MyGraph
	# mk		# Max number of groups
	# k		# Number of groups
	# g		# Group assignment
	# n		# Group sizes
	# m		# Edge count (between/within) groups

	def __init__(self,G,k,maxk):


		self._G = G
		self._k = k
		self._mk = maxk
		self.__initGroupAssignment()
		self.__initGroupSize()
		self.__initEdgeCount()



	''' update k -> k+1'''
	def kPlusOne(self):
		self._k = min(self._k+1, self._mk)

	''' update k -> k-1'''
	def  kMinusOne(self):

		if any(self._n[:self._k]==0):

			emp_list = np.where(self._n[:self._k]==0)[0]

			rk = np.random.choice(emp_list)

			self._k -= 1

			# update g
			self._g[self._g == self._k] = rk

			# update n
			self._n[rk] = self._n[self._k]
			self._n[self._k] = 0

			# update m
			self._m[rk,:self._k] = self._m[self._k,:self._k]
			self._m[:self._k,rk] = self._m[:self._k,self._k]
			self._m[rk,rk] = self._m[self._k,self._k]
			self._m[self._k,:] = 0
			self._m[:,self._k] = 0
			



	''' Node i moves from group r to s '''
	def move(self,r,s,i): 

		if r==s: return
		
		d = [ np.sum( self._g[self._G._nb[i]] == rr ) for rr in range(self._k) ]

		self._m[r,:self._k] -= d
		self._m[:self._k,r] -= d
		self._m[s,:self._k] += d
		self._m[:self._k,s] += d

		self._n[r] -= 1
		self._n[s] += 1

		self._g[i] = s




	''' Private field '''
	def __initGroupAssignment(self):
		self._g = np.random.randint(0, self._k, self._G._N)

	def __initGroupSize(self):
		self._n = np.zeros(self._mk, dtype='int')
		self._n[:self._k] = np.array( [ np.sum( self._g == kk) for kk in range(self._k)] )
	
	def __initEdgeCount(self):
		self._m = np.zeros((self._mk,self._mk),dtype='int')
		self._m[:self._k,:self._k] = np.array( [[ np.sum(  np.prod( (self._g[self._G._dl])==[r,s] ,axis=1)  ) for s in range(self._k)] for r in range(self._k)] )

