
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
import seaborn as sns
from sklearn import manifold
from sklearn.cluster import KMeans

from my_functions import *


# import colour
sns.set_style('white')

LL = .6
SS = .8


"""

Divergences

"""
def get_divergence(p, q, option):
	p = p/np.sum(p)
	q = q/np.sum(q)
	if option == 1:
		return total_variation(p,q)
	elif option == 2:
		return l2_norm(p,q)
	elif option == 3:
		return kl(p,q)
	elif option == 4:
		return js(p,q)
	elif option == 5:
		return pearson_chi_square(p,q)
	elif option == 6:
		return neyman_chi_square(p,q)
	elif option == 7:
		return rukhin(p,q)
	elif option == 8:
		return triangular_discrimination(p,q)
	elif option == 9:
		return squared_hellinger(p,q)
	elif option == 10:
		return matsusita(p,q)
	elif option == 11:
		return piuri_vinche(p,q)
	elif option == 12:
		return arimono(p,q)
	else:
		print('unknown divergence')
		sys.exit()


def total_variation(p, q):
	return np.sum(np.abs(p-q))
def l2_norm(p, q):
	return np.sum((p-q)**2)
def kl(p, q):
	return np.sum(  p * np.log( (p+10**(-10))/(q+10**(-10)) )  )
def js(p, q):
	return (kl(p, q)+kl(q, p))/2
def pearson_chi_square(p, q):
	return np.sum(  (  ((p-q)**2)/(q+10**(-10))  )  )
def neyman_chi_square(p, q):
	return np.sum(  (  ((p-q)**2)/(p+10**(-10))  )  )
def rukhin(p, q, a=0.3):
	return np.sum(  (  ((p-q)**2)/( (1-a)*p+a*q+10**(-10) )  )  )
def triangular_discrimination(p, q):
	return np.sum(  (  ((p-q)**2)/( p+q+10**(-10) )  )  )
def squared_hellinger(p, q):
	return np.sum(  ( np.sqrt(p)-np.sqrt(q) )**2  )
def matsusita(p, q, a=0.3):
	return np.sum(  np.abs( p**a-q**a )**(1./a)  )
def piuri_vinche(p, q, a=3):
	return np.sum(  np.abs( p-q )**a / ( (p+q)**(a-1)+10**(-10) )  )
def arimono(p, q):
	return np.sum(  np.sqrt(p**2+q**2) - (p+q)/np.sqrt(2)  )



"""

Manifold learning

"""
def tsne(X, perplexity, n_componets):
	Y = manifold.TSNE(n_componets, perplexity=perplexity).fit_transform(X)
	order = np.argsort(Y.reshape(1,-1)[0])
	return order, Y

def isomap(X,n_neighbors,n_componets=1):
	Y = manifold.Isomap(n_neighbors, n_componets).fit_transform(X)
	order = np.argsort(Y.reshape(1,-1)[0])
	return order, Y



"""

kmeans

"""
def similarity_matrix(X, option):
	N = X.shape[0]
	M0 = X.shape[1]
	sim_mx = np.zeros((M0, M0))
	for m1 in range(M0):
		for m2 in range(M0):
			sim_mx[m1, m2] = get_divergence(X[:,m1], X[:,m2], option)
	return sim_mx

def kmeans(sim_mx, n_clusters):
	labels = KMeans(n_clusters=n_clusters).fit_predict(sim_mx)
	return labels

def plot_2d(grp_assing_filename, cfg):
	X = preprocessing(grp_assing_filename)
	labels, Y = dim_reduction(X, cfg.n_clusters, cfg.divergence_option, cfg.perplexity)  # labels: labels from kmeans, Y: N_by_2 matrix from t-SNE

	color_pallete, cmap_1d = my_color_pallete_1d(cfg.n_clusters)
	str_mk = "o^sdh"
	plt.figure(figsize=cfg.figsize)
	for i in range(cfg.n_clusters):
		boolean = labels == i
		plt.scatter(Y[:,0][boolean], Y[:,1][boolean], s=cfg.markersize, marker=str_mk[int(i/len(str_mk))], color=color_pallete[i], alpha=0.7)
		
	if cfg.is_savefig:
		if not os.path.exists(cfg.save_path):
			os.makedirs(cfg.save_path)
		plt.savefig(cfg.save_path+cfg.fig_filename)

	plt.show()

	return X, labels

def dim_reduction(X, n_clusters, divergence_option, perplexity):
	sim_mx = similarity_matrix(X, divergence_option)
	labels = kmeans(sim_mx, n_clusters)
	order, Y = tsne(sim_mx, perplexity, 2)
	# order, Y = isomap(sim_mx, 3, 2)

	return labels, Y


"""

palette diagram

"""

def palette_diagram(X, labels, cfg):
	X0 = get_representative_assignments(X, labels, cfg.n_clusters)
	table, (X0, order) = _palette_diagram(X0, cfg.n_neighbors, cfg.figsize, cfg.is_savefig, cfg.save_path, cfg.fig_filename)
	record_parameters(cfg, cfg.save_path)

	return table


def _palette_diagram(X, n_neighbors, figsize=(12,9), is_savefig=False, save_path='../figures/', filename='default.png'):

	X, order = manifold_learning(X, n_neighbors)

	color_pallete, cmap_1d = my_color_pallete_1d(X.shape[1])
	color_matrix = my_color_palette_2d(X, cmap_1d)

	fig, (ax1,ax2) = plt.subplots(2,1,sharex=True, figsize=figsize)
	ax1.stackplot(range(X.shape[0]), X.T, baseline='wiggle', colors=color_pallete, alpha=0.7)

	ax2.imshow(color_matrix, alpha=0.7)
	ax2.set_aspect('auto')
	

	ax1.grid(linestyle='--')
	ax1.xaxis.set_minor_locator(tick.AutoMinorLocator())

	ax2.grid(linestyle='--')
	ax2.xaxis.set_minor_locator(tick.AutoMinorLocator())
	ax2.yaxis.set_major_locator(tick.MaxNLocator(integer=True))

	if is_savefig:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		plt.savefig(save_path+filename)

	plt.show()

	

	table = np.zeros((order.size, 2))
	for i,ind in enumerate(order):
		table[i,0] = i
		table[i,1] = ind

	return table, (X, order)


def manifold_learning(data, n_neighbors):

	n = data.shape[0]
	if n < n_neighbors:
		n_neighbors = int(n/2)

	order, Y = isomap(data, n_neighbors)

	return data[order,:], order

		
def my_color_palette_2d(data, cmap_1d):

	N = data.shape[0]
	M = data.shape[1]

	color_palette_2d = []

	for m in range(M):
		row = []
		for n in range(N):
			elm = data[n,m]
			ll = 1.0 - 0.4*elm
			palette = sns.hls_palette(1, h=cmap_1d[m], l=ll, s=SS)[0]
			row.append(palette)

		color_palette_2d.append(row)

	return color_palette_2d


def my_color_pallete_1d(M):

	cmap_1d = np.linspace(0, 1, M+1)[:-1]

	color_palette = []
	for sim in cmap_1d:
		color_palette.append( sns.hls_palette(1, h=sim, l=LL, s=SS)[0] )
		
	return color_palette, cmap_1d


def entropy(p_d):
	p_d = p_d/np.sum(p_d)
	ent = 0
	for p in p_d:
		if p==0: continue
		ent -= p*np.log(p)
	return ent

def get_min_entropy(X):
	n = X.shape[0] # Number of prob distributions
	d = X.shape[1] # Dimension of prob distributions

	min_ent = 100
	min_ent_ind = 0
	for i in range(n):
		ent = entropy(X[i,:])
		if min_ent > ent:
			min_ent = ent
			min_ent_ind = i
	return X[min_ent_ind,:]

def get_max_entropy(X):
	n = X.shape[0] # Number of prob distributions
	d = X.shape[1] # Dimension of prob distributions

	max_ent = 100
	max_ent_ind = 0
	for i in range(n):
		ent = entropy(X[i,:])
		if max_ent < ent:
			max_ent = ent
			max_ent_ind = i
	return X[max_ent_ind,:]

def get_median_entropy(X):
	n = X.shape[0] # Number of prob distributions
	d = X.shape[1] # Dimension of prob distributions

	X_soated = sorted(X, key=lambda x: entropy(x))
	return X_soated[int(n/2)]

def get_max_area(X):
	n = X.shape[0] # Number of prob distributions
	d = X.shape[1] # Dimension of prob distributions

	max_area = 0
	max_area_ind = 0
	for i in range(n):
		area = np.sum(X[i,:])
		if max_area < area:
			max_area = area
			max_area_ind = i
	return X[max_area_ind,:]

def get_random_entropy(X):
	r = np.random.randint(X.shape[0])
	return X[r, :]


def get_representative_assignments(X, labels, n_clusters):
	X0 = np.zeros((X.shape[0], n_clusters))
	for cluster in range(n_clusters):
		boolean = labels == cluster
		X_cluster = X[:, boolean]
		# X0[:,cluster] = get_max_entropy(X_cluster.T)
		# X0[:,cluster] = get_random_entropy(X_cluster.T)
		X0[:,cluster] = get_max_area(X_cluster.T)
	return X0


"""

palette diagrams

"""

def palette_diagrams(X_list, order_list, labels_list, figsize, is_savefig=False, save_path='../figures/palette_diagrams/', filename='default.png'):


	n_X = len(X_list)
	fig, ax = plt.subplots(n_X, 1, sharex=True, figsize=figsize)
	fig.subplots_adjust(hspace=0)

	X1 = X_list[0]
	order1 = order_list[0]
	for i in range(n_X):

		X = X_list[i]
		order = order_list[i]
		color_pallete1, cmap_1d1 = my_color_pallete_1d(X.shape[1])

		if i != 0:

			order_X = np.zeros((X.shape[0],2), dtype='int')
			order_X[:,0] = np.arange(X.shape[0])
			order_X[:,1] = order
			order_X = order_X[order_X[:,1].argsort(),:]

			ind_list = list(range(X.shape[0]))
			ind2_list = order_X[order1[ ind_list ], 0]

			X = X[ind2_list,:]


		ax[i].stackplot(range(X.shape[0]), X.T, baseline='wiggle', colors=color_pallete1, alpha=0.7)
		ax[i].grid(linestyle='--')
		ax[i].xaxis.set_minor_locator(tick.AutoMinorLocator())
		ax[i].set_yticklabels([])

	if is_savefig:
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		plt.savefig(save_path+filename)

	plt.show()




"""

preprocessing

"""
def order_to_stack(pg_list):
	pgs = []
	for pg in pg_list:
		for m in range(pg.shape[1]):
			pgs.append(pg[:,m])
	pgs = np.array(pgs).T
	return pgs


def read_file(filename):
	pg_list = []
	f = open(filename, 'r')
	N = int(f.readline())
	rows = []
	line = 0
	for row in f:
		line+=1
		row = row.replace('\n', '')
		row = row.split(' ')
		row = list(map(float, row))
		rows.append(row)
		if line%N==0:
			pg_list.append( np.array(rows) )
			rows = []
	return pg_list


def eliminate_all_zero_partitions(data):
	k = data.shape[1]
	boolean = np.ones(k, dtype='bool')
	for i in range(k):
		if not data[:,i].any():
			boolean[i] = False
	data = data[:,boolean]
	return data


def eliminate_single_partition(pg_list):
	pg_list = [pd for pd in pg_list if pd.shape[1] > 1]
	return pg_list


def preprocessing(grp_assing_filename):
	pg_list = read_file(grp_assing_filename)
	# pg_list = eliminate_single_partition(pg_list)
	raw_data = order_to_stack(pg_list)
	raw_data = eliminate_all_zero_partitions(raw_data)
	return raw_data





