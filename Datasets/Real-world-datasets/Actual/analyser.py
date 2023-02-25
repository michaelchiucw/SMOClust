
# analyser.py
# Copyright (C) 2022 University of Birmingham, Birmingham, United Kingdom
# @author Chun Wai Chiu (cxc1015@student.bham.ac.uk)

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# he Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.


# Usage: python3 analyser.py -m [minorityClassLabel] -b [blockSize] -f [stream.arff] > [result.txt]

# Assuming class attribute is the last attribute
import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys

import arff
import pandas as pd
import itertools
import numpy as np

from statistics import mean
from statistics import stdev
from statistics import median

from sklearn.neighbors import KDTree
from sklearn.cluster import AffinityPropagation

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.size': 22})

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

SAFE = "SAFE"
BORDERLINE = "BORDER"
RARE = "RARE"
OUTLIER = "OUTLIER"
ONLY_MAJORITY = "Only majority"
TYPES = [SAFE, BORDERLINE, RARE, OUTLIER, ONLY_MAJORITY]

RAND_SEED = [13, 53, 101, 151, 199, 263, 317, 383, 443, 503, 577, 641, 701, 769, 839, 911, 983, 1049, 1109, 1193, 1277, 1321, 1429, 1487, 1559, 1619, 1699, 1783, 1871, 1949]

def main(args):
	fileName = args.f
	if not os.path.exists(fileName):
		sys.exit(fileName + " not found")

	BLOCK_SIZE = int(args.b)
	min_index = int(args.m)

	print("File Name:", fileName)
	print("BLOCK_SIZE:", BLOCK_SIZE)
	print("min_index:", min_index)

	# load arff file
	file = open(fileName)
	file_dict = arff.load(file)

	# build data frame
	attributes_list = file_dict['attributes']
	attribute_names = [ name for (name,values) in  attributes_list ]
	data_list = file_dict['data']

	df = pd.DataFrame(data_list, columns=attribute_names)

	# get list of possible class values
	class_list = [ values for (name,values) in  attributes_list if name == 'class' ][0]

	# preprocessing: one hot encoding nominal attributes
	col_to_convert = []
	for (name, values) in attributes_list:
		if values != 'NUMERIC':
			df[name] = pd.Categorical(df[name], categories=values)
			if name != 'class':
				col_to_convert.append(name)

	df = pd.get_dummies(df, columns=col_to_convert)

	# variables for recording statistics
	start_index = 0
	end_index = BLOCK_SIZE

	class_ratios_batches = [ [] for x in class_list ]


	num_of_min = []
	numOfClusters = []

	safe_ratios = []
	borderline_ratios = []
	rare_ratios = []
	outlier_ratios = []

	batch_id = 0
	batch_id_min = []

	num_of_uncounted_batches = 0
	num_of_batches = 0

	plot_num_of_min = []
	plot_numOfClusters = []

	plot_safe_ratios = []
	plot_borderline_ratios = []
	plot_rare_ratios = []
	plot_outlier_ratios = []

	plot_xticks = []

	while end_index <= len(df):
		batch = df.iloc[start_index:end_index]

		# Estimate ratios for the classes of the current batch
		all_class_ratios = computeIR(batch, class_list)
		for i in range(len(class_ratios_batches)):
			class_ratios_batches[i].append(all_class_ratios[i])

		# if all_class_ratios[min_index] * BLOCK_SIZE > 0:
		plot_num_of_min.append(all_class_ratios[min_index] * BLOCK_SIZE)
		if all_class_ratios[min_index] * BLOCK_SIZE >= 6:
			batch_id_min.append(batch_id)
			num_of_min.append(all_class_ratios[min_index] * BLOCK_SIZE)

			# Esitmate the potential number of clusters
			num_of_clusters = estimateClusters(batch, class_list, min_index)
			numOfClusters.append(num_of_clusters)
			plot_numOfClusters.append(num_of_clusters)

			# Esitmate the Minority class types
			types_count_result = countMinorityClassTypes(batch, class_list, min_index)

			n_s = batch['class'].value_counts()
			min_label = class_list[min_index]
			min_n = float(n_s[min_label])
			
			safe_ratios.append(0 if min_n == 0 else (float(types_count_result[SAFE]) / min_n))
			borderline_ratios.append(0 if min_n == 0 else (float(types_count_result[BORDERLINE]) / min_n))
			rare_ratios.append(0 if min_n == 0 else (float(types_count_result[RARE]) / min_n))
			outlier_ratios.append(0 if min_n == 0 else (float(types_count_result[OUTLIER]) / min_n))

			plot_safe_ratios.append(0 if min_n == 0 else (float(types_count_result[SAFE]) / min_n))
			plot_borderline_ratios.append(0 if min_n == 0 else (float(types_count_result[BORDERLINE]) / min_n))
			plot_rare_ratios.append(0 if min_n == 0 else (float(types_count_result[RARE]) / min_n))
			plot_outlier_ratios.append(0 if min_n == 0 else (float(types_count_result[OUTLIER]) / min_n))
		else:
			num_of_uncounted_batches = num_of_uncounted_batches + 1

			plot_numOfClusters.append(0)

			plot_safe_ratios.append(0)
			plot_borderline_ratios.append(0)
			plot_rare_ratios.append(0)
			plot_outlier_ratios.append(0)


		num_of_batches = num_of_batches + 1

		plot_xticks.append(end_index)

		if end_index >= len(data_list):
			break
		else:
			start_index = start_index + BLOCK_SIZE
			end_index = min(end_index + BLOCK_SIZE, len(data_list))
			batch_id = batch_id + 1



	print("Total number of examples: " + str(len(data_list)))
	for class_index in range(len(class_list)):
		print("\nClass " + str(class_index) + " ratio:")
		print(printStats(class_ratios_batches[class_index]))

	print("\nEstimated minority clusters:")
	print(printStats(numOfClusters))
	
	print("\nSafe ratio:")
	print(printStats(safe_ratios))

	print("\nBorderline ratio:")
	print(printStats(borderline_ratios))

	print("\nRare ratio:")
	print(printStats(rare_ratios))

	print("\nOutlier ratio:")
	print(printStats(outlier_ratios))

	print("\nLatex table row:")
	print(printLatexTableRow(len(data_list), num_of_batches, num_of_uncounted_batches, numOfClusters, class_ratios_batches[min_index], safe_ratios, borderline_ratios, rare_ratios, outlier_ratios))

	print("\nplot_xticks:")
	print(plot_xticks)

	plotMinRatios(class_ratios_batches[min_index], plot_xticks, fileName)
	plotNumClusters(plot_numOfClusters, plot_xticks, fileName)
	plotMinorityTypeRatios(plot_safe_ratios, plot_borderline_ratios, plot_rare_ratios, plot_outlier_ratios, plot_xticks, fileName)

	# print("Minority Type ratio by batch:")
	# d = {'batch id':batch_id_min,'Est. #Clusters':numOfClusters, '#Minority':num_of_min, 'Safe':safe_ratios, 'Borderline':borderline_ratios, 'Rare':rare_ratios, 'Outlier':outlier_ratios}
	# mt_df = pd.DataFrame(data=d)
	# print(mt_df)

	file.close()

def estimateClusters(df, class_list, min_index, random_state=1):
	df_copy = df.reset_index(drop=True)
	min_label = class_list[min_index]
	df_min = df_copy[df_copy['class'] == min_label]
	df_min = df_min.reset_index(drop=True)
	df_min = df_min.drop(columns=['class'])

	if len(df_min) < 6:
		return 0

	af_clustering_result = AffinityPropagation(copy=True, affinity='euclidean', random_state=random_state).fit(df_min)

	cluster_labels_df = pd.DataFrame(af_clustering_result.labels_, columns=['cluster'])
	cluster_count = cluster_labels_df['cluster'].value_counts()

	valid_cluster_count = [ (cluster, count) for cluster, count in cluster_count.items() if count >= 6 ]

	return len(valid_cluster_count)

	# count_per_run = []

	# for seed in RAND_SEED:
	# 	af_clustering_result = AffinityPropagation(copy=True, affinity='euclidean', random_state=random_state).fit(df_min)

	# 	cluster_labels_df = pd.DataFrame(af_clustering_result.labels_, columns=['cluster'])
	# 	cluster_count = cluster_labels_df['cluster'].value_counts()

	# 	valid_cluster_count = [ (cluster, count) for cluster, count in cluster_count.items() if count >= 6 ]

	# 	count_per_run.append(len(valid_cluster_count))

	# return mean(count_per_run)


def countMinorityClassTypes(df, class_list, min_index):
	safety = detect_types_napierala(df, get_mudict(0., 0., class_list, [0.]), class_list, k=5)

	mapped = list(map(get_label_from_probability, safety[min_index], [ False for x in safety[min_index] ]))
	mapped_df = pd.DataFrame(mapped, columns=['types'])
	mapped_df['types'] = pd.Categorical(mapped_df['types'], categories=TYPES)
	
	return mapped_df.value_counts()

def get_mudict(between_min, between_min_maj, u_labels, maj):
	dict={}
	for i,j in itertools.product(u_labels, u_labels):
		# print("i:", i, "| j:", j)
		if i == j :
			dict[(i,j)] = 1.0
		elif any([k in maj for k in [i,j]]):
			dict[(i,j)] = between_min_maj
		else:
			dict[(i,j)] = between_min
	return dict

def detect_types_napierala(df, mu_dict, class_list, k=5):
	df_copy = df.reset_index(drop=True)
	dataset = df_copy.iloc[:, df_copy.columns != 'class'].copy()
	labels = df_copy['class']
	kdt = KDTree(dataset, leaf_size=20, metric='euclidean')
	nn = kdt.query(dataset, k=k+1, return_distance=False)
	safe_level_by_class = [ [] for i in range(len(class_list)) ]
	for i in range(len(nn)):
		knn = set(nn[i]) - set([i])
		assert len(knn) == k
		neighbours = [(labels[i], labels[j]) for j in knn]
		safety = [ mu_dict[n] for n in neighbours ]
		current_class_index = class_list.index(labels[i])
		safe_level_by_class[current_class_index].append(float(sum(safety)) / float(k))
	return safe_level_by_class

# def detect_types_napierala(df, mu_dict,  k=5):
# 	df_copy = df.reset_index(drop=True)
# 	dataset = df_copy.iloc[:, df_copy.columns != 'class'].copy()
# 	labels = df_copy['class']
# 	kdt = KDTree(dataset, leaf_size=20, metric='euclidean')
# 	nn = kdt.query(dataset, k=k+1, return_distance=False)
# 	safe_level = list()
# 	for i in range(len(nn)):
# 		knn = set(nn[i]) - set([i])
# 		assert len(knn) == k
# 		neighbours = [(labels[i], labels[j]) for j in knn]
# 		safety = [ mu_dict[n] for n in neighbours ]
# 		safe_level.append(float(sum(safety)) / float(k))
# 	return safe_level

def get_label_from_probability(prob, only_majority=True):
	if prob > 0.7:
		return SAFE
	elif prob > 0.3:
		return BORDERLINE
	elif prob > 0.1:
		return RARE
	elif prob > 0:
		return OUTLIER
	else:
		if only_majority:
			return ONLY_MAJORITY
		else:
			return OUTLIER

def computeIR(df, class_list):
	total_inst = len(df)

	all_class_counts = df['class'].value_counts()

	all_class_ratios = []
	for label in class_list:
		all_class_ratios.append(all_class_counts[label] / total_inst)

	# min_ratio = min(all_class_ratios)
	# min_index = all_class_ratios.index(min_ratio)

	return all_class_ratios

def listQuartiles(xs):
	sorted_list = sorted(xs)
	med = median(sorted_list)
	mid = len(sorted_list) // 2
	if (len(sorted_list) % 2 == 0):
		# even
		lowerQ = median(sorted_list[:mid])
		upperQ = median(sorted_list[mid:])
	else:
		# odd
		lowerQ = median(sorted_list[:mid])  # same as even
		upperQ = median(sorted_list[mid+1:])

	return lowerQ, med, upperQ

def printStats(xs):
	# print(xs)
	# print("xs len:",len(xs))
	to_print = str(round(min(xs),3)) + " - " + str(round(max(xs),3)) + "\n"
	to_print += "Mean: " + str(round(mean(xs),3)) + " | Std. Dev.: " + str(round(stdev(xs),3)) + "\n"
	q1, q2, q3 = listQuartiles(xs)
	iqr = q3 - q1
	to_print += "q1, q2, q3: " + str(round(q1,3)) + ", " + str(round(q2,3)) + ", " + str(round(q3,3)) + " | iqr: " + str(round(iqr,3))
	# to_print += "q1, q2, q3: " + str(q1) + ", " + str(q2) + ", " + str(q3) + " | iqr: " + str(iqr)

	return to_print

def printLatexTableRow(numExamples, numBatches, numUncountedBatches, numOfClusters, min_ratios, safe_ratios, borderline_ratios, rare_ratios, outlier_ratios):
	to_print = str(numExamples) + " & "
	to_print += str(numBatches) + " (" + str(numUncountedBatches) + ") & "
	to_print += printCellWithMedian(numOfClusters, isRatio=False) + " & "
	to_print += printCellWithMedian(min_ratios, isRatio=True) + " & "
	to_print += printCellWithMedian(safe_ratios, isRatio=True) + " & "
	to_print += printCellWithMedian(borderline_ratios, isRatio=True) + " & "
	to_print += printCellWithMedian(rare_ratios, isRatio=True) + " & "
	to_print += printCellWithMedian(outlier_ratios, isRatio=True)

	return to_print

def printCellWithMedian(xs, isRatio=True):
	to_print = ""
	if isRatio:
		l = [ x * 100 for x in xs ]
		to_print += str(round(min(l))) + "\\%-" + str(round(max(l))) + "\\% (" + str(round(median(l))) + "\\%)"
	else:
		to_print += str(round(min(xs))) + "-" + str(round(max(xs))) + " (" + str(round(median(xs))) + ")"

	return to_print

def plotMinRatios(min_ratios, xticks, fileName):
	x = xticks
	y = [ ratio*100 for ratio in min_ratios ]

	xMax = x[len(x)-1]

	plt.plot(x,y,
		color='r',
		linewidth=3.5)

	plt.xlabel('TimeSteps')
	plt.ylabel('Minority Class Ratio (%)')

	plt.xlim(0,int(xMax))
	plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

	plt.ylim(0,100)

	plt.grid(True)

	imageName = "./analysis-plots/" + fileName.replace(".arff", "-minority-ratios.png")
	plt.savefig(imageName, bbox_inches='tight', dpi = 100)
	plt.clf()

def plotNumClusters(numOfClusters, xticks, fileName):
	x = xticks
	y = numOfClusters

	xMax = x[len(x)-1]
	yMax = max(y) + 1
	yMin = max(min(y) - 1, 0)

	plt.plot(x,y,
		color='r',
		linewidth=3.5)

	plt.xlabel('TimeSteps')
	plt.ylabel('#Minority Class Sub-Clusters')

	plt.xlim(0,int(xMax))
	plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

	# plt.ylim(yMin,yMax)

	plt.grid(True)

	imageName = "./analysis-plots/" + fileName.replace(".arff", "-num-of-clusters.png")
	plt.savefig(imageName, bbox_inches='tight', dpi = 100)
	plt.clf()

def plotMinorityTypeRatios(safe_ratios, borderline_ratios, rare_ratios, outlier_ratios, xticks, fileName):
	x = xticks
	ys = [safe_ratios, borderline_ratios, rare_ratios, outlier_ratios]

	xMax = x[len(x)-1]

	colors = ['green', 'orange', 'red', 'black']
	labels = ['Safe', 'Borderline', 'Rare', 'Outlier']
	for y, color, lb in zip(ys, colors, labels):
		y_to_plot = [ percent*100 for percent in y ]
		plt.plot(x,y_to_plot,
			label=lb,
			color=color,
			linewidth=3.5)

	plt.xlabel('TimeSteps')
	plt.ylabel('Minority Example Type (%)')

	plt.ylim(0, 100)
	plt.xlim(0,int(xMax))
	plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", fontsize=22, borderaxespad=0.)

	plt.grid(True)
	imageName = "./analysis-plots/" + fileName.replace(".arff", "-minority-type.png")
	plt.savefig(imageName, bbox_inches='tight', dpi = 100)
	plt.clf()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-m', help="minority class index", action="store")
	parser.add_argument('-b', help="block size", action="store")
	parser.add_argument('-f', help="File name (.arff)", action="store")
	args = parser.parse_args()
	main(args)