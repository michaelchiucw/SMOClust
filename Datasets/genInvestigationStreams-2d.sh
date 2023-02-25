#!/bin/bash

# genInvestigationStreams-2d.sh
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

ACTUAL_DEST="$BASEDIR/ActualStreams-investigation"
ACTUAL_TRAIN_DEST="$ACTUAL_DEST/Training-Actual-investigation"
ACTUAL_TEST_DEST="$ACTUAL_DEST/Testing-Actual-investigation"

mkdir -p $ACTUAL_DEST
mkdir -p $ACTUAL_TRAIN_DEST
mkdir -p $ACTUAL_TEST_DEST

JAR_FILE_NAME="moa-2018.6.1-SNAPSHOT-project-dec-bound-2d.jar"

MODEL_SEED=(567)
TRAIN_SEED=(123)
TEST_SEED=(890)

NUM_OF_ATTRIBUTES=2
TOTAL_INST=200000
EVAL_INTERVAL=500
TEST_SET_SIZE=500

IR=(0.3 0.1 0.03 0.01 0.003)
CLUSTERS=(7)


# Output .arff file format:
# Train: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TRAIN_SEED[seed_index]}-train.arff  
# Test: 	{Name}-ms=${MODEL_SEED[seed_index]}-is=${TEST_SEED[seed_index]}-test_{i}.arff 

# ======================================================================

for seed_index in {0..0}; do
	#---------------------------------------Single Factor--------------------------------------
	for ir in "${IR[@]}"; do
		if [[ $ir < 0.01 ]]; then
			ir_str=$(python3 -c "print(0,int($ir * 1000),sep=\"\")")
		else
			ir_str=$(python3 -c "print(int($ir * 100))")
		fi
		for cluster in "${CLUSTERS[@]}"; do
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
		done

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