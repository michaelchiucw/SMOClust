#!/bin/bash

# genStreams-severely-imbalanced-5d.sh
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

ACTUAL_DEST="$BASEDIR/SynthStream_DataDifficulty/ActualStreams-severely-imbalanced-review"
ACTUAL_TRAIN_DEST="$ACTUAL_DEST/Training-Actual-severely-imbalanced-review"
ACTUAL_TEST_DEST="$ACTUAL_DEST/Testing-Actual-severely-imbalanced-review"

mkdir -p $ACTUAL_DEST
mkdir -p $ACTUAL_TRAIN_DEST
mkdir -p $ACTUAL_TEST_DEST

JAR_FILE_NAME="moa-2018.6.1-SNAPSHOT-project-dec-bound-2d.jar"

MODEL_SEED=(19 61 107 163 223 271 337 397 457 521 593 647 719 787 857 929 997 1061 1123 1213 1283 1361 1439 1493 1571 1627 1721 1789 1877 1973)
TRAIN_SEED=(5 41 83 137 191 241 307 367 431 487 563 617 677 751 823 883 967 1031 1093 1171 1237 1303 1409 1471 1543 1607 1669 1753 1847 1913)
TEST_SEED=(23 67 109 167 227 277 347 401 461 523 599 653 727 797 859 937 1009 1063 1129 1217 1289 1367 1447 1499 1579 1637 1723 1801 1879 1979)

NUM_OF_ATTRIBUTES=2
TOTAL_INST=200000
EVAL_INTERVAL=500
TEST_SET_SIZE=500

IR=(0.05 0.03 0.01 0.007 0.005 0.003)
CLUSTERS=(3 7)


# Output .arff file format:
# Train: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff  
# Test: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test_{i}.arff 

# ======================================================================

for seed_index in {0..29}; do
	#---------------------------------------Single Factor--------------------------------------
	for ir in "${IR[@]}"; do
		if [[ $ir < 0.01 ]]; then
			ir_str=$(python3 -c "print(0,int($ir * 1000),sep=\"\")")
		else
			ir_str=$(python3 -c "print(int($ir * 100))")
		fi
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
done

exit 0