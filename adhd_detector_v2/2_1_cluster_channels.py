import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys



sys.path.append("/home/gari/Desktop/master_tesis_v3/")

from OFHandlers import OFHandlers as OFH




############################################################################
def GSN_HydroCel_129_to_df(path_cvs_channel_loc):
	"""
	get dataframe from csv for GSN_HydroCel_129 montage
	"""

	#REf How context influences the interpretation of facial expressions: a source
	#localization high-density EEG study on the “Kuleshov effect”
	exclude_outermost_channels=["E43", "E48", "E49", 
	                            "E56", "E63", "E68",
	                            "E73", "E81", "E88",
	                            "E94", "E99","E107", 
	                            "E113", "E119", "E120", 
	                            "E125", "E126", "E127", 
	                            "E128"]
	#read coordinates of the montage
	channels_pos=pd.read_csv(path_cvs_channel_loc)[["labels","X","Y","Z"]]

	#remove channels from montage in exclude_outermost_channels
	channels_pos=channels_pos[~channels_pos["labels"].isin(exclude_outermost_channels)]
	channels_pos.set_index("labels",inplace=True)

	return channels_pos




############################################################################
def standard_10_20_to_df(path_cvs_channel_loc):
	"""
	get dataframe from csv for standard_10_20 montage
	"""

	#read coordinates of the montage
	channels_pos=pd.read_csv(path_cvs_channel_loc)[["labels","X","Y","Z"]]
	channels_pos.set_index("labels",inplace=True)
	return channels_pos




############################################################################
def clustering_channels(channels_pos,n_clusters):
	"""
	Return dataframe of -channels_pos 	-cluster 	X 	Y 	Z
	"""
	#Initialize clustering
	clustering=KMeans(n_clusters=n_clusters,
					init='k-means++',
					max_iter=300,
					n_init=1000,
					random_state=0)
	clustering.fit(channels_pos)
	#get centers
	centers=clustering.cluster_centers_

	print("centers in \n", centers)

	#calculate aritmetic sum of componements
	# id_cluster:sum_compo
	r={index:(each_center[0]+each_center[1]+each_center[2]) for 
							index,each_center in enumerate(centers)}

	#get cluster sorted id_cluster:smallest sum of components
	r_sorted=sorted(r.items(), key=lambda x: x[1])

	#get id of clusters
	new_order_keys=[each[0] for each in r_sorted]

	#make a copy of the centriods
	centers_unsorted=centers.copy()

	#re assigne the clusters centers
	for index,each_sorted_key in enumerate(new_order_keys):
	    clustering.cluster_centers_[index]=centers_unsorted[each_sorted_key]

	print("centers out \n", clustering.cluster_centers_)

	channels_clusters=pd.DataFrame.from_dict({'channels_pos':channels_pos.index,
										'cluster':clustering.predict(channels_pos),
                                        'X':channels_pos.X,
                                        'Y':channels_pos.Y,
                                        'Z':channels_pos.Z})
	return channels_clusters




############################################################################
def plot_save_clustered_montage(channels_clusters,
								n_clusters,
								save=True,
								path_to_save="./"):
	"""
	Sava clustered montage image
	"""

	colors_list=sns.color_palette("husl", n_clusters)
	dictionary_channels={}
	i=0
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(111, projection='3d')
	for label,df in channels_clusters.groupby("cluster"):
	    print("***********")
	    print(label)
	    print(df)
	    #if(label==7 or label==6):
	    ax.scatter(df.X, df.Y, df.Z,color=colors_list[i],label=label)
	    ax.plot_trisurf(df.X, df.Y, df.Z,color=colors_list[i],alpha=0.5)
	    ax.view_init(30)
	    #ax.view_init(30)
	    i=i+1
	plt.legend() 
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	

	if save==True:
		plt.savefig(path_to_save,
					pad_inches=0.0,
					transparent=True,
					bbox_inches="tight")




############################################################################
if __name__== "__main__":

	#csv file channel loc
	path_to_data="/home/gari/Desktop/master_tesis_v3/Data/"


	#number of cluster to create
	n_clusters=5


	##########for GSN_HydroCel_129_chanlocs
	path=path_to_data+"MontageForClustering/GSN_HydroCel_129_chanlocs.csv"
	path_to_save_img=path_to_data+"MontageForClustering/GSN_HydroCel_129_clustered.png"
	path_to_save_df=path_to_data+"MontageForClustering/GSN_HydroCel_129_clustered.file"

	channels_pos=GSN_HydroCel_129_to_df(path_cvs_channel_loc=path)

	channels_clusters=clustering_channels(channels_pos=channels_pos,
											n_clusters=n_clusters)

	OFH.save_object(path_to_save_df,channels_clusters)

	plot_save_clustered_montage(channels_clusters=channels_clusters,
								n_clusters=n_clusters,
								save=True,
								path_to_save=path_to_save_img)

	

	##########for 10_20_chanlocs
	path=path_to_data+"MontageForClustering/10_20_chanlocs.csv"
	path_to_save_img=path_to_data+"MontageForClustering/standard_10_20_to_clustered.png"
	path_to_save_df=path_to_data+"MontageForClustering/standard_10_20_to_clustered.file"

	channels_pos=standard_10_20_to_df(path_cvs_channel_loc=path)

	channels_clusters=clustering_channels(channels_pos=channels_pos,
											n_clusters=n_clusters)

	plot_save_clustered_montage(channels_clusters=channels_clusters,
								n_clusters=n_clusters,
								save=True,
								path_to_save=path_to_save_img)

	OFH.save_object(path_to_save_df,channels_clusters)

		