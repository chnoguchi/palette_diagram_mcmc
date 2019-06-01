Palette diagram
====

A visualization tool for graph clustering

## Description
Palette diagrams allow you to assess the group assignment distributions obtained as results of graph clustering algorithms.

## Prerequisites
* numpy 1.14.5
* matplotlib 2.2.2
* scikit-learn 0.19.1
* seaborn 0.8.1
* easydict 1.9
* python-igraph 0.7.1

## Usage
This code consists of two steps:
1. Inference of group assignments
2. Construction of the palette diagrams

To draw a palette diagram, you first perform graph clustering to obtain the group assignment distributions.
The MCMC algorithm `[1]` (which we implemented in python) can be replaced by another clustering algorithm. 

### 1. Inference of group assignments (MCMC)
* Input  
An edgelist file in the GML format (0-based indexing). Its location must be specified by `edgelist_filename`.  
* The detail of parameters for MCMC
  * `K` - Initial number of groups, which also indicates maximum number of groups.  
  * `relaxation_time` - Until MCMC sweep count reaches this number, MCMC sampling is not performed.  
  * `sample_interval` - Interval MCMC sampling is performed.   
  * `n_samples` - Number of times MCMC sampling is performed. 
  * `gml_filename` - Relative path to the GML file and the filename.
* Output  
A text file in which the group assingment distributions are written. The first raw represents the number of vertices in the given network. 
The rest of the raws represent `n_samples` group assingment distributions, so that the number of raws of the outputs file is M = n_samples * N + 1, where N is the number of vertices. The output file is saved as `./newman2016/outputs/grp_assign.txt`.

> The mcmc algorithm here is for a demonstration purpose. We recommend you to replace it with the original C++ implementation `[1]` when you analyze large datasets. 


### 2. Palette diagrams  
This step consists of two substeps. 
#### 2.1 kmeans & t-SNE  
The objective of this step is to determine the number of groups and the color assignments of those groups. 
To this end, a clustering algorithm (kmeans) and a manifold learning algorithm (t-SNE) are performed. 
This step is to be repeated manually until an appropriate parameters is found.
* Input  
A text file written in the same format as the output file of MCMC. Its location must be specified by `grp_assing_filename`.  
* The detail of parameters
  * `perplexity` - Parameter for t-SNE. This is related to the variance of probability distribution of the data. Higher variance and larger data usually require a larger perplexity. The detail of perplexity is explained at [here](https://distill.pub/2016/misread-tsne/) or the original paper of t-SNE [2].
  * `n_clusters` - Number of clusters identified by the kmeans.
  * `divergenve_option` - Option to determine the similarity measure to calculate the similarity matrix.
    * **1** - Total variation distance
    * **2** - L2 norm
    * **3** - KL divergence
    * **4** - JS divergence
    * **5** - Pearson Chi-square distance
    * **6** - Neyman Chi square
    * **7** - Rukhin distance
    * **8** - Triangular discrimination
    * **9** - Squared Hellinger distance
    * **10** - Matsusita distance
    * **11** - Piuri and Vinche divergence
    * **12** - Arimoto distance
  * `is_savefig` - Save figure or not.
  * `save_path` - Relative path where the output figure is to be saved.
  * `fig_filename` - Filename of the output figure.
  * `grp_assing_filename` - Relative path to the input file and the filename (This parameter is needed only when this second step is performed independently).
* Output  
A scatter plot in two-dimensional space, shown in the jupyter notebook, and cluster labels given by the kmeans, stored in the python local variable `labels`.
#### 2.2 Palette diagram  
This step draws the one-dimensional and two-dimensional palette diagrams.
* Input  
The cluster labels given by the kmeans, which is output of previous cell.
* The detail of parameters
  * `n_neighbors` - Number of neighbors to calculate an approximate geodesic distance for the Isomap.
  * `figsize` - Size of the output figure.
  * `fig_filename` - Filename of the figure to be save.
* Output  
The palette diagram is saved in `save_path` as a PDF.

## Reference  
[1] M. E. J. Newman and G. Reinert, Phys. Rev. Lett. 117, 078301 (2016). 
The original code (implemented in C++) is available at [here](http://www-personal.umich.edu/~mejn/).  
[2] L. J. P. van der Maaten and G.E. Hinton, Journal of Machine Learning Research 9:2579-2605 (2008).

