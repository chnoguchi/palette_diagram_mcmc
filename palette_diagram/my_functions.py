import numpy as np
import pickle
import os


def record_parameters(cfg, save_path):

	# Record parameters
	with open(save_path+'PARAMETERS_%d.txt' % cfg.n_clusters, 'w') as f:
	   f.write("::::::::::::::: ALGORITHM :::::::::::::::\n")
	   if cfg.algorithm_option == 1:
	       f.write("EM algorithm : Decelle et al. (2011)\n")
	   elif cfg.algorithm_option == 2:
	       f.write("EM algorithm in Julia v0.6\n")
	   elif cfg.algorithm_option == 3:
	       f.write("Algorithm = MCMC Newman-Reinert (2016)\n")
	       f.write("Maximum and initial number of groups = "+str(cfg.K)+"\n")
	       f.write("Relaxation time = "+str(cfg.relaxation_time)+"\n")
	       f.write("Sample interval = "+str(cfg.sample_interval)+"\n")
	       f.write("Number of samples = "+str(cfg.n_samples)+"\n")
	   elif cfg.algorithm_option == 4:
	       f.write("graph-tool (overlapping SBM)\n")
	   else:
	       f.write("UNKNOWN algorithm\n")

	   f.write("\n\n")
	   f.write("::::::::::::::: PALETTE DIAGRAM :::::::::::::::\n")
	   f.write("== Cluster Selection (2-dim t-SNE embedding) ==\n")
	   f.write("perplexity = "+str(cfg.perplexity)+"\n")
	   f.write("n_clusters = "+str(cfg.n_clusters)+"\n")
	   f.write("divergenve_option = "+str(cfg.divergence_option)+"\n")

	   f.write("\n")
	   f.write("== Palette Diagram (1-dim ISOMAP embedding) ==\n")
	   f.write("n_neighbors = "+str(cfg.n_neighbors)+" (Number of neighbors to contruct a weighted graph for isomap.)\n")



