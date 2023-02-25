#!/bin/bash

# genPreStream.sh
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

PRE_DEST="$BASEDIR/SynthStream_DataDifficulty/PreStreams"
PRE_TRAIN_DEST="$PRE_DEST/Training-Pre"
PRE_TEST_DEST="$PRE_DEST/Testing-Pre"

mkdir -p $PRE_DEST
mkdir -p $PRE_TRAIN_DEST
mkdir -p $PRE_TEST_DEST

MODEL_SEED=(19 61 107 163 223 271 337 397 457 521)
PRE_TRAIN_SEED=(7 43 89 139 193 251 311 373 433 491)
PRE_TEST_SEED=(29 71 113 173 229 281 349 409 463 541)

JAR_FILE_NAME="moa-2018.6.1-SNAPSHOT-project-dec-bound-2d.jar"

NUM_OF_ATTRIBUTES=2
TOTAL_INST=200000
EVAL_INTERVAL=500
TEST_SET_SIZE=500

DRIFT_TYPE=("Im"
	"Borderline"
	"Rare"
	"Split/Move/Merge")

# Initialise the random seed for bash environment.
PARAM_RAND_SEED=6313
echo "Random seed: $PARAM_RAND_SEED"
python3 --version
RANDOM=$PARAM_RAND_SEED

# [INIT_CLUSTERS, INIT_MINORITY_SHARE, NUM_OF_DRIFT_TYPE, ]
python3_cmd="import random; \
	random.seed($PARAM_RAND_SEED); \
	print(random.randint(1,7)); \
	print(random.uniform(0.0, 0.5)); \
	print(random.randint(1,${#DRIFT_TYPE[@]})); \
	"

values=($(python3 -c "$python3_cmd"))


INIT_CLUSTERS=${values[0]}
INIT_MINORITY_SHARE=${values[1]}
INIT_SAFE=1.000000
INIT_BORDERLINE=0.000000
INIT_RARE=0.000000
NUM_OF_DRIFT_TYPE=${values[2]}

tmp_TRAIN_DESC=""

for (( i=0; i<$NUM_OF_DRIFT_TYPE; i++ )); do
	# get index
	drift_type_index=$(( RANDOM % ${#DRIFT_TYPE[@]} ))
	# get drift type
	drift_type=${DRIFT_TYPE[drift_type_index]}
	# remove chosen drift type from array
	unset DRIFT_TYPE[drift_type_index]
	# reindex the array
	DRIFT_TYPE=( "${DRIFT_TYPE[@]}" )

	tmp_TRAIN_DESC+="${drift_type}:"
done

TRAIN_DESC=""
TEST_DESC=""
drift_type=""

#===============================Split/Move/Merg===============================
if [[ $tmp_TRAIN_DESC == *"Split/Move/Merge"* ]]; then

	Split_Move_Merge=("Split" "Move" "Merge")
	drift_type=${Split_Move_Merge[$(( RANDOM % ${#Split_Move_Merge[@]} ))]}

	if [[ $drift_type == "Split" ]]; then

		tmp_str="splitting-clusters/sudden,start=1,end=1,value-start=0,value-end=0:"
		tmp_str+="clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:"
		tmp_str+="clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:"

		TRAIN_DESC="$tmp_str$TRAIN_DESC"
		TEST_DESC="$tmp_str$TEST_DESC"

	elif [[ $drift_type == "Move" ]]; then

		tmp_str="clusters-movement/incremental,start=70000,end=85000,value-start=0,value-end=1:"
		tmp_str+="clusters-movement/incremental,start=85000,end=100000,value-start=0,value-end=1:"

		TRAIN_DESC+=$tmp_str
		TEST_DESC+=$tmp_str

	elif [[ $drift_type == "Merge" ]]; then

		TRAIN_DESC+="splitting-clusters/incremental,start=70000,end=100000,value-start=1,value-end=0:"
		TEST_DESC+="splitting-clusters/incremental,start=70000,end=100000,value-start=1,value-end=0:"

	fi

	tmp_TRAIN_DESC="${tmp_TRAIN_DESC/Split\/Move\/Merge/$drift_type}"
fi

if [[ $drift_type == "Move" || $drift_type == "Merge" ]]; then
	tmp_TRAIN_DESC="${tmp_TRAIN_DESC//Borderline:}"
	tmp_TRAIN_DESC="${tmp_TRAIN_DESC//Rare:}"
fi

echo "=========="
echo -e "tmp_TRAIN_DESC:\n${tmp_TRAIN_DESC}"
all_details+="${tmp_TRAIN_DESC}\n\n"
echo "=========="

#====================================Im=====================================
if [[ $tmp_TRAIN_DESC == *"Im"* ]]; then
	TARGET_MINORITY_SHARES=(0.5 0.4 0.3 0.2 0.1 0.05 0.03 0.02 0.01)
	target_minority_share=${TARGET_MINORITY_SHARES[$(( RANDOM % ${#TARGET_MINORITY_SHARES[@]} ))]}

	echo "target_minority_share: ${target_minority_share}"
	all_details+="target_minority_share: ${target_minority_share}\n"

	end_value=$(python3 -c "print($target_minority_share / $INIT_MINORITY_SHARE)")

	TRAIN_DESC+="minority-share/incremental,start=70000,end=100000,value-start=1,value-end=$end_value:"
fi

#===============================Borderline,Rare===============================
if [[ $drift_type != "Move" && $drift_type != "Merge" ]]; then
	if [[ $tmp_TRAIN_DESC == *"Borderline"* && $tmp_TRAIN_DESC == *"Rare"* ]]; then

		INIT_BORDERLINE=0.000001
		INIT_RARE=0.000001

		TARGET=(0.2 0.4 0.6)
		target_borderline=${TARGET[$(( RANDOM % ${#TARGET[@]} ))]}
		target_rare=$(python3 -c "print(0.8-$target_borderline)")

		echo "target_borderline: ${target_borderline}"
		all_details+="target_borderline: ${target_borderline}\n"

		echo "target_rare: ${target_rare}"
		all_details+="target_rare: ${target_rare}\n"

		end_value_borderline=$(python3 -c "print(($INIT_SAFE*$target_borderline) / ((1-$target_borderline-$target_rare)*$INIT_BORDERLINE))")
		end_value_rare=$(python3 -c "print(($INIT_BORDERLINE*$end_value_borderline*$target_rare) / ($target_borderline*$INIT_RARE))")

		TRAIN_DESC+="borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=$end_value_borderline:"
		TRAIN_DESC+="rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=$end_value_rare:"

	elif [[ $tmp_TRAIN_DESC == *"Borderline"* ]]; then

		INIT_BORDERLINE=0.000001

		TARGER_BORDERLINE=(0.2 0.4 0.6 0.8 1.0)
		target_borderline=${TARGER_BORDERLINE[$(( RANDOM % ${#TARGER_BORDERLINE[@]} ))]}

		echo "target_borderline: ${target_borderline}"
		all_details+="target_borderline: ${target_borderline}\n"

		end_value=$(python3 -c "print(($INIT_SAFE*$target_borderline) / ( $INIT_BORDERLINE*(1-$target_borderline)))")

		TRAIN_DESC+="borderline-ratio/incremental,start=70000,end=100000,value-start=1,value-end=$end_value:"

	elif [[ $tmp_TRAIN_DESC == *"Rare"* ]]; then

		INIT_RARE=0.000001

		TARGER_RARE=(0.2 0.4 0.6 0.8 1.0)
		target_rare=${TARGER_RARE[$(( RANDOM % ${#TARGER_RARE[@]} ))]}

		echo "target_rare: ${target_rare}"
		all_details+="target_rare: ${target_rare}\n"

		end_value=$(python3 -c "print(($INIT_SAFE*$target_rare) / ( $INIT_RARE*(1-$target_rare)))")

		TRAIN_DESC+="rare-ratio/incremental,start=70000,end=100000,value-start=1,value-end=$end_value:"

	fi
fi

echo "=========="

TRAIN_DESC=${TRAIN_DESC%:}
TEST_DESC=${TEST_DESC%:}

echo -e "TRAIN_DESC:\n${TRAIN_DESC}"
all_details+="\nTRAIN_DESC:\n${TRAIN_DESC}\n\n"

echo -e "TEST_DESC:\n${TEST_DESC}"
all_details+="\nTEST_DESC:\n${TEST_DESC}\n\n"

echo "=========="

echo "INIT_CLUSTERS: ${INIT_CLUSTERS}"
echo "INIT_MINORITY_SHARE: ${INIT_MINORITY_SHARE}"
echo "INIT_SAFE: ${INIT_SAFE}"
echo "INIT_BORDERLINE: ${INIT_BORDERLINE}"
echo "INIT_RARE: ${INIT_RARE}"
echo "NUM_OF_DRIFT_TYPE: ${NUM_OF_DRIFT_TYPE}"

all_details+="INIT_CLUSTERS: ${INIT_CLUSTERS}\n"
all_details+="INIT_MINORITY_SHARE: ${INIT_MINORITY_SHARE}\n"
all_details+="INIT_SAFE: ${INIT_SAFE}\n"
all_details+="INIT_BORDERLINE: ${INIT_BORDERLINE}\n"
all_details+="INIT_RARE: ${INIT_RARE}\n"
all_details+="NUM_OF_DRIFT_TYPE: ${NUM_OF_DRIFT_TYPE}\n\n"

mkdir -p $PRE_DEST
mkdir -p $PRE_TRAIN_DEST
mkdir -p $PRE_TEST_DEST
rm -f $PRE_TRAIN_DEST/* $PRE_TEST_DEST/*
echo -e $all_details > $PRE_TRAIN_DEST/pre-stream-details.txt

# Output .arff file format:
# Train: 	Pre-ms=${MODEL_SEED[seed_index]}-is=${PRE_TRAIN_SEED[seed_index]}-train.arff 
# Test: 	Pre-ms=${MODEL_SEED[seed_index]}-is=${PRE_TEST_SEED[seed_index]}-test_{i}.arff 

# ======================================================================

for seed_index in {0..9}; do
	java -cp "$BASEDIR/moa-release-2018.6.0/$JAR_FILE_NAME:$BASEDIR/moa-release-2018.6.0/lib/*" -javaagent:$BASEDIR/moa-release-2018.6.0/lib/sizeofag-1.0.4.jar moa.DoTask \
		"WriteTrainingStreamAndTestSetsToARFF -s 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${PRE_TRAIN_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n $INIT_CLUSTERS -m $INIT_MINORITY_SHARE -s $INIT_SAFE -b $INIT_BORDERLINE -p $INIT_RARE -o 0 -u 
				-d '$TRAIN_DESC') 
			-t 
			(generators.ImbalancedDriftGenerator -r ${MODEL_SEED[seed_index]} -i ${PRE_TEST_SEED[seed_index]} -a $NUM_OF_ATTRIBUTES 
				-n $INIT_CLUSTERS -m 0.5 -s $INIT_SAFE -b $INIT_BORDERLINE -p $INIT_RARE -o 0 -u 
				-d '$TEST_DESC') 
			-b 
			-j $PRE_TRAIN_DEST/Pre-ms=${MODEL_SEED[seed_index]}-is=${PRE_TRAIN_SEED[seed_index]}-train.arff 
			-k $PRE_TEST_DEST/Pre-ms=${MODEL_SEED[seed_index]}-is=${PRE_TEST_SEED[seed_index]}-test.arff 
			-x $TEST_SET_SIZE 
			-p $EVAL_INTERVAL 
			-m $TOTAL_INST"
done

exit 0