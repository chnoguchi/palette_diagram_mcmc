from igraph import Graph
import numpy as np
from scipy.special import gammaln


class MyGraph:
	# N				# Number of nodes
	# l				# Pair of nodes(i<j)

	def getNumberOfLinks(self):
		return self._l.shape[0]

	def initMembers(self, N, links):
		self._N = N
		self._l = links
		self._dl = np.concatenate((self._l, np.fliplr(self._l)))
		self._nb = [self._dl[self._dl[:,0]==lnode,1].tolist() for lnode in range(self._N)]

class Network:

	# G		# MyGraph
	# mk	# Max number of groups
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


def log_factorial(n):
	return gammaln(n+1)

def logp_gk(G,network):
	k = network._k
	n = G._N
	nr = network._n[:k]
	return np.sum(  log_factorial(nr)  )

def logp_Ag(G,network):
	k = network._k
	n = G._N
	m = network._m[:k,:k]
	nr = network._n[:k]
	kap = np.array( [ np.sum( m[r,:] ) for r in range(k)] )
	p = 2.0*G.getNumberOfLinks()/(n**2)
	nzn = nr[nr > 0]
	nzkap = kap[nr > 0]
	# print nr
	# print kap

	# return np.sum(  log_factorial(np.diag(m)) - ( np.diag(m)+1 )*np.log( p*(nr**2)/2.+1 )  ) \
	# + np.sum(  log_factorial(m) - (m+1)*np.log( p*np.dot(nr.reshape(k,1), nr.reshape(1,k) )+1 )  )/2. \
	# - np.sum(  log_factorial(np.diag(m)) - ( np.diag(m)+1 )*np.log( p*(nr**2)+1 )  )/2.

	return np.sum(  log_factorial(np.diag(m)/2) - ( np.diag(m)/2+1 )*np.log( p*(nr**2)/2.+1 )  ) \
	+ np.sum(  log_factorial(nzn-1) - log_factorial(nzn+nzkap-1) + nzkap*np.log(nzn)  ) \
	+ np.sum(  log_factorial(m) - (m+1)*np.log( p*np.dot(nr.reshape(k,1), nr.reshape(1,k) )+1 )  )/2. \
	- np.sum(  log_factorial(np.diag(m)) - ( np.diag(m)+1 )*np.log( p*(nr**2)+1 )  )/2.


def update_k(G, network):
	k = network._k
	n = G._N

	# k -> k+1
	if np.random.rand() < 0.5:

		if np.random.rand() < float(k)/(n+k):
			network.kPlusOne()


	# k -> k-1
	else:
		network.kMinusOne()


def prob_gi(G,network,ni):

	gi = network._g[ni]

	E = logp_Ag(G,network) + logp_gk(G,network)
	newE = np.zeros(network._k)

	for r in range(network._k):

		if gi == r:
			newE[r] = E
		else:
			network.move(gi,r,ni)
			newE[r] = logp_Ag(G,network) + logp_gk(G,network)
			network.move(r,gi,ni)

	return (newE,E)

def prob_g(G,network):

	pg = np.zeros((G._N, network._k))
	for ni in range(G._N):
		newE,E = prob_gi(G,network,ni)
		pg[ni,:] = bp = np.exp(newE-E) / np.sum( np.exp(newE-E) )

	return pg

	

def mcmc(G,network,max_swp,eq_swp=0):

	group_hist = np.zeros(network._mk, dtype='int')

	for swp in range(max_swp):

		for cnt in range(G._N):

			if np.random.rand() < 1.0/(G._N+1): update_k(G,network)


			k = network._k
			n = G._N
			ni = np.random.randint(n)
			gi = network._g[ni]

			newE,E = prob_gi(G,network,ni)

			bp = np.exp(newE-E) / np.sum( np.exp(newE-E) )
			s = np.random.choice(range(k), p=bp)

			network.move(gi,s,ni)

		if swp > eq_swp: 
			group_hist[network._k-1] += 1

		# print( 'k:%d, sweep:%d, logp:%f' % (network._k, swp, newE[s]) )

	return group_hist

def read_gml(filename):
	g = Graph.Read_GML(filename)
	edgelist = g.get_edgelist()
	data = np.zeros((len(edgelist), 2), dtype=np.int)
	for i, edge in enumerate(edgelist):
		data[i, 0] = edge[0]
		data[i, 1] = edge[1]
	return data


def exec_mcmc(gml_filename, sample_interval, n_samples, relaxation_time, K):
	data = read_gml(gml_filename)

	N = max( max(data[:,0]), max(data[:,1]) )
	N = N + 1
	print('N=%d' % N)

	G = MyGraph()
	G.initMembers(N,data)

	network = Network(G,K,K)

	pg_list = []
	for sample in range(n_samples):
		group_hist = mcmc(G, network, sample_interval, relaxation_time)
		pg = prob_g(G,network)
		pg_list.append(pg)

	out_filename = "newman2016/outputs/grp_assign.txt"
	with open(out_filename, 'w') as f:
		f.write(str(N))
		f.write('\n')
		for pg in pg_list:

			with open(out_filename, 'a'):
				for i in range(pg.shape[0]):
					for j in range(pg.shape[1]):
						f.write('{:.6f}'.format(pg[i,j]))
						if j == pg.shape[1]-1:
							break
						f.write(' ')
					f.write('\n')
