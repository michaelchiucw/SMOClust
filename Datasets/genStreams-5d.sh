#!/bin/bash

# genStreams-5d.sh
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

BASEDIR="" # <-- Put your directory to this folder here.

ACTUAL_DEST="$BASEDIR/SynthStream_DataDifficulty/ActualStreams-review"
ACTUAL_TRAIN_DEST="$ACTUAL_DEST/Training-Actual-review"
ACTUAL_TEST_DEST="$ACTUAL_DEST/Testing-Actual-review"

mkdir -p $ACTUAL_DEST
mkdir -p $ACTUAL_TRAIN_DEST
mkdir -p $ACTUAL_TEST_DEST

JAR_FILE_NAME="moa-2018.6.1-SNAPSHOT-project-dec-bound-5d.jar"

MODEL_SEED=(13 53 101 151 199 263 317 383 443 503 577 641 701 769 839 911 983 1049 1109 1193 1277 1321 1429 1487 1559 1619 1699 1783 1871 1949)
TRAIN_SEED=(1 2 31 73 127 179 233 283 353 419 467 547 607 661 739 811 877 947 1019 1087 1153 1229 1297 1381 1453 1523 1597 1663 1741 1823)
TEST_SEED=(29 71 113 173 229 281 349 409 463 541 601 659 733 809 863 941 1013 1069 1151 1223 1291 1373 1451 1511 1583 1657 1733 1811 1889 1987)

NUM_OF_ATTRIBUTES=5
TOTAL_INST=200000
EVAL_INTERVAL=500
TEST_SET_SIZE=500

IR=(0.3 0.1 0.01)
CLUSTERS=(3 7)


# Output .arff file format:
# Train: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff  
# Test: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test_{i}.arff 

# ======================================================================

for seed_index in {0..29}; do
	#---------------------------------------Single Factor--------------------------------------
	for ir in "${IR[@]}"; do
		ir_str=$(python3 -c "print(int($ir * 100))")
		for cluster in "${CLUSTERS[@]}"; do
			# StaticIm$ir_Split{3,7}
			java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
				"WriteTrainingStreamAndTestSetsToARFF -s 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m $ir -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
					-t 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
					-b 
					-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Split$cluster-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
					-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Split$cluster-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
					-x $TEST_SET_SIZE 
					-p $EVAL_INTERVAL 
					-m $TOTAL_INST"
			# (( counter++ ))
			# StaticIm$ir_Move{3,7}
			java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
				"WriteTrainingStreamAndTestSetsToARFF -s 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m $ir -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
					-t 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
					-b 
					-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Move$cluster-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
					-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Move$cluster-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
					-x $TEST_SET_SIZE 
					-p $EVAL_INTERVAL 
					-m $TOTAL_INST"
			# (( counter++ ))

			# StaticIm$ir_Merge{3,7}
			java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
				"WriteTrainingStreamAndTestSetsToARFF -s 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m $ir -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'splitting-clusters/incremental,start=70000,end=100000,value-start=1,value-end=0') 
					-t 
					(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
						-n $cluster -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
						-d 'splitting-clusters/incremental,start=70000,end=100000,value-start=1,value-end=0') 
					-b 
					-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Merge$cluster-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
					-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Merge$cluster-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
					-x $TEST_SET_SIZE 
					-p $EVAL_INTERVAL 
					-m $TOTAL_INST"
			# (( counter++ ))
		done

		# Borderline20
		java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
			"WriteTrainingStreamAndTestSetsToARFF -s 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m $ir -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=250000.000000') 
				-t 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
				-b 
				-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Borderline20-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
				-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Borderline20-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
				-x $TEST_SET_SIZE 
				-p $EVAL_INTERVAL 
				-m $TOTAL_INST"
		# (( counter++ ))

		# Borderline100
		java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
			"WriteTrainingStreamAndTestSetsToARFF -s 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m $ir -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
				-t 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
				-b 
				-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
				-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
				-x $TEST_SET_SIZE 
				-p $EVAL_INTERVAL 
				-m $TOTAL_INST"
		# (( counter++ ))

		# Rare20
		java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
			"WriteTrainingStreamAndTestSetsToARFF -s 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m $ir -s 1.000000 -b 0.000000 -p 0.000001 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=250000.000000') 
				-t 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
				-b 
				-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Rare20-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
				-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Rare20-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
				-x $TEST_SET_SIZE 
				-p $EVAL_INTERVAL 
				-m $TOTAL_INST"
		# (( counter++ ))

		# Rare100
		java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
			"WriteTrainingStreamAndTestSetsToARFF -s 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m $ir -s 1.000000 -b 0.000000 -p 0.000001 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
				-t 
				(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
					-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
					-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
				-b 
				-j $ACTUAL_TRAIN_DEST/StaticIm$ir_str-Rare100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
				-k $ACTUAL_TEST_DEST/StaticIm$ir_str-Rare100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
				-x $TEST_SET_SIZE 
				-p $EVAL_INTERVAL 
				-m $TOTAL_INST"
		# (( counter++ ))
	done

	# StaticIm10_Im1 {10%->1%}
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.100000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.100000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d '') 
			-b 
			-j $ACTUAL_TRAIN_DEST/StaticIm10_Im1-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/StaticIm10_Im1-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# StaticIm1_Im10 {1%->10%}
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.010000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'minority-share/incremental,start=70000,end=100000,value-start=1,value-end=10.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d '') 
			-b 
			-j $ACTUAL_TRAIN_DEST/StaticIm1_Im10-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/StaticIm1_Im10-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Im1 {50%->1%}
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.020000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d '') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im1-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im1-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# StaticIm1_Im50 {1%->50%}
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.010000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'minority-share/incremental,start=70000,end=100000,value-start=1,value-end=50.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 1 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d '') 
			-b 
			-j $ACTUAL_TRAIN_DEST/StaticIm1_Im50-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/StaticIm1_Im50-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	#---------------------------------------Double Factor---------------------------------------

	# Im1+Rare100
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000001 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.020000:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im1+Rare100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im1+Rare100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Im10+Rare60
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000001 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=1500000.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im10+Rare60-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im10+Rare60-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Split5+Im10
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Split5+Im10-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Split5+Im10-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Im1+Borderline100
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.020000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im1+Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im1+Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Im10+Borderline20
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=250000.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im10+Borderline20-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im10+Borderline20-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	#---------------------------------------Complex Factor--------------------------------------

	# StaticIm10_Split5+Im1+Rare100
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.100000 -s 1.000000 -b 0.000000 -p 0.000001 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.100000:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
			-b 
			-j $ACTUAL_TRAIN_DEST/StaticIm10_Split5+Im1+Rare100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/StaticIm10_Split5+Im1+Rare100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# StaticIm10_Split5+Im1+Borderline100
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.100000 -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.100000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=999998999971.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
			-b 
			-j $ACTUAL_TRAIN_DEST/StaticIm10_Split5+Im1+Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/StaticIm10_Split5+Im1+Borderline100-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Split5+Im10+Borderline40+Rare40
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000001 -p 0.000001 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=2000000.000000:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=2000000.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Split5+Im10+Borderline40+Rare40-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Split5+Im10+Borderline40+Rare40-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Split5+Im10+Borderline80
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000001 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=4000000.000000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Split5+Im10+Borderline80-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Split5+Im10+Borderline80-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))

	# Im10+Borderline20+Rare20
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000001 -p 0.000001 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:minority-share/incremental,start=70000,end=100000,value-start=1,value-end=0.200000:borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=333333.500000:rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=333333.500000') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n 5 -m 0.500000 -s 1.000000 -b 0.000000 -p 0.000000 -o 0 -u 
				-d 'splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0') 
			-b 
			-j $ACTUAL_TRAIN_DEST/Im10+Borderline20+Rare20-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff 
			-k $ACTUAL_TEST_DEST/Im10+Borderline20+Rare20-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
	# (( counter++ ))
done

exit 0