/*
 *    WriteTrainingAndTestSetToARFF.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.tasks;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.Writer;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.options.ClassOption;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

import moa.streams.ExampleStream;
import moa.streams.generators.ImbalancedDriftGenerator;


public class WriteTrainingStreamAndTestSetsToARFF extends AuxiliarMainTask {

    @Override
    public String getPurposeString() {
        return "Outputs a stream to an ARFF file.";
    }

    private static final long serialVersionUID = 1L;
    
//    public IntOption randomSeedOption = new IntOption(
//            "randomSeed", 'i',
//            "Seed for weighted random selection of past Borderline, Rare, and Outliner instances.", 1);
    
    public ClassOption trainingStreamOption = new ClassOption("trainingStream", 's',
            "Training stream to write.", ExampleStream.class,
            "generators.ImbalancedDriftGenerator");
    
    public ClassOption testingStreamOption = new ClassOption("testingStream", 't',
            "Testing stream to write.", ExampleStream.class,
            "generators.ImbalancedDriftGenerator");
    
    public FlagOption balanceTestSetOption = new FlagOption("balancedTestSet", 'b',
    		"Make sure the Test Sets are balanced?");

    public FileOption trainingARFFFileOption = new FileOption("trainingARFFFile", 'j',
            "Destination training ARFF file.", null, "arff", true);
    
    public FileOption testingARFFFileOption = new FileOption("testingARFFFile", 'k',
            "Destination testing ARFF file.", null, "arff", true);
    
    public IntOption testSetSizeOption = new IntOption("testSetSize", 'x',
    		"Size of each test set.", 500, 0, Integer.MAX_VALUE);
    
    public IntOption evaluationIntervalOption = new IntOption("evaluationInterval", 'p',
    		"Time step interval to pause the stream progress and generate test sets.", 500, 0, Integer.MAX_VALUE);

    public IntOption maxInstancesOption = new IntOption("maxInstances", 'm',
            "Maximum number of instances to write to file.", 200000, 0,
            Integer.MAX_VALUE);

    public FlagOption suppressHeaderOption = new FlagOption("suppressHeader",
            'h', "Suppress header from output.");
    
    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
//    	Random random = new Random(this.randomSeedOption.getValue());
    	
    	ImbalancedDriftGenerator trainingStream = (ImbalancedDriftGenerator) this.getPreparedClassOption(this.trainingStreamOption);
    	ImbalancedDriftGenerator testingStream = (ImbalancedDriftGenerator) this.getPreparedClassOption(this.testingStreamOption);
    	
    	if (this.balanceTestSetOption.isSet()) {
    		testingStream.positiveShareOption.setValue(0.5);
    	}
//    	this.printStreamsState(trainingStream, testingStream);
    	
    	String testSetFileCLI = this.testingARFFFileOption.getValueAsCLIString();
    	int testSetCounter = 0;
    	
    	File trainingStreamFile = this.trainingARFFFileOption.getFile();
    	if (trainingStreamFile != null) {
    		try {
    			Writer trainingInstWriter = new BufferedWriter(new FileWriter(trainingStreamFile));
    			monitor.setCurrentActivityDescription("Writing training stream to ARFF");
    			if (!this.suppressHeaderOption.isSet()) {
                    trainingInstWriter.write(trainingStream.getHeader().toString());
                    trainingInstWriter.write("\n");
                }
    			int trainingInstWritten = 0;
    			
    			LinkedList<Instance> borderlineInsts = new LinkedList<Instance>();
    			LinkedList<Integer> borderlineTimeStamps = new LinkedList<Integer>();
    			
                LinkedList<Instance> rareInsts = new LinkedList<Instance>();
                LinkedList<Integer> rareTimeStamps = new LinkedList<Integer>();
                
                LinkedList<Instance> outlinerInsts = new LinkedList<Instance>();
                LinkedList<Integer> outlinerTimeStamps = new LinkedList<Integer>();
                
                while ((trainingInstWritten < this.maxInstancesOption.getValue())
                        && trainingStream.hasMoreInstances()) {
                	// Write training stream to ARFF
                	Instance currentTrainingInst = trainingStream.nextInstance().getData();
                    trainingInstWriter.write(currentTrainingInst.toString());
                    trainingInstWriter.write("\n");
                    trainingInstWritten++;
                    // Let the testing stream to go through as well.
                    testingStream.nextInstance();
                    
                    // Storing Borderline, Rare, Outliner respectively.
                    if (currentTrainingInst.classValue() == 1) {
                    	int currentPositiveType = trainingStream.getCurrentPositiveType();
                    	switch(currentPositiveType) {
	                		case 0:
	                			// Safe case, do nothing.
	                			break;
	                		case 1:
	                			// Borderline
	                			borderlineInsts.add(currentTrainingInst);
	                			borderlineTimeStamps.add(trainingInstWritten);
	                			break;
	                		case 2:
	                			// Rare
	                			rareInsts.add(currentTrainingInst);
	                			rareTimeStamps.add(trainingInstWritten);
	                			break;
	                		case 3:
	                			// Outliner
	                			outlinerInsts.add(currentTrainingInst);
	                			outlinerTimeStamps.add(trainingInstWritten);
	                			break;
	            			default:
	            				throw new IllegalArgumentException("Unknown instance type: " + currentPositiveType);
	                	}
                    }
                    
                    // Write test sets
                    if (trainingInstWritten % this.evaluationIntervalOption.getValue() == 0) {
                    	monitor.setCurrentActivityDescription("Pause writing training stream to ARFF");
                    	
                    	// Convert lists to arrays
                    	Iterator<Instance> inst_itr;
                    	Iterator<Integer> timeStamp_itr;
                    	int itr_index;
                    	
                    	
                    	// Borderline
                    	Instance[] borderlineInsts_array = new Instance[borderlineInsts.size()];
                    	double[] borderlineWeights = new double[borderlineTimeStamps.size()];

                    	inst_itr = borderlineInsts.iterator();
                    	timeStamp_itr = borderlineTimeStamps.iterator();
                    	itr_index = 0;
                    	
                    	while (inst_itr.hasNext() && timeStamp_itr.hasNext()) {
                    		borderlineInsts_array[itr_index] = inst_itr.next().copy();
                    		borderlineWeights[itr_index] = timeStamp_itr.next().intValue();
                    		++itr_index;
                    	}
                    	
                    	// Rare
                    	Instance[] rareInsts_array = new Instance[rareInsts.size()];
                    	double[] rareWeights = new double[rareTimeStamps.size()];

                    	inst_itr = rareInsts.iterator();
                    	timeStamp_itr = rareTimeStamps.iterator();
                    	itr_index = 0;
                    	
                    	while (inst_itr.hasNext() && timeStamp_itr.hasNext()) {
                    		rareInsts_array[itr_index] = inst_itr.next().copy();
                    		rareWeights[itr_index] = timeStamp_itr.next().intValue();
                    		++itr_index;
                    	}

                    	// Outliner
                    	Instance[] outlinerInsts_array = new Instance[outlinerInsts.size()];
                    	double[] outlinerWeights = new double[outlinerTimeStamps.size()];

                    	inst_itr = outlinerInsts.iterator();
                    	timeStamp_itr = outlinerTimeStamps.iterator();
                    	itr_index = 0;
                    	
                    	while (inst_itr.hasNext() && timeStamp_itr.hasNext()) {
                    		outlinerInsts_array[itr_index] = inst_itr.next().copy();
                    		outlinerWeights[itr_index] = timeStamp_itr.next().intValue();
                    		++itr_index;
                    	}
                    	
                    	// Weighted random selection of past Borderline, Rare, and Outliner instances.
                    	int numOfBorderline = borderlineInsts_array.length == 0 ? 0 : (int) (trainingStream.getCurrentBorderlinePortion() * this.testSetSizeOption.getValue() / 2.0);
                    	int numOfRare = rareInsts_array.length == 0 ? 0 : (int) (trainingStream.getCurrentRarePortion() * this.testSetSizeOption.getValue() / 2.0);
                    	int numOfOutliner = outlinerInsts_array.length == 0 ? 0 : (int) (trainingStream.getCurrentOutlierPortion() * this.testSetSizeOption.getValue() / 2.0);
                    	
//                    	System.out.println("borderlineInsts size: " + borderlineInsts.size());
//                    	System.out.println("rareInsts size: " + rareInsts.size());
//                    	System.out.println("outlinerInsts size: " + outlinerInsts.size());
//                    	
//                    	System.out.println("\nBorderline Portion: " + trainingStream.getCurrentBorderlinePortion());
//                    	System.out.println("Rare Portion: " + trainingStream.getCurrentRarePortion());
//                    	System.out.println("Outliner Portion: " + trainingStream.getCurrentOutlierPortion());
//                    	
//                    	System.out.println("\nnumOfBorderline: " + numOfBorderline);
//                    	System.out.println("numOfRare: " + numOfRare);
//                    	System.out.println("numOfOutliner: " + numOfOutliner);
                    	
                    	Instance[] testing_Borderline = new Instance[numOfBorderline];
                    	Instance[] testing_Rare = new Instance[numOfRare];
                    	Instance[] testing_Outliner = new Instance[numOfOutliner];
                    	
                    	for (int i = 0; i < testing_Borderline.length; ++i) {
                    		int index = MiscUtils.chooseRandomIndexBasedOnWeights(borderlineWeights, testingStream.getInstanceRandom());
//                    		testing_Borderline[i] = borderlineInsts_array[index];
                    		testing_Borderline[i] = this.addNoise(borderlineInsts_array[index], testingStream.getInstanceRandom());
                    	}
                    	for (int i = 0; i < testing_Rare.length; ++i) {
                    		int index = MiscUtils.chooseRandomIndexBasedOnWeights(rareWeights, testingStream.getInstanceRandom());
//                    		testing_Rare[i] = rareInsts_array[index];
                    		testing_Rare[i] = this.addNoise(rareInsts_array[index], testingStream.getInstanceRandom());
                    	}
                    	for (int i = 0; i < testing_Outliner.length; ++i) {
                    		int index = MiscUtils.chooseRandomIndexBasedOnWeights(outlinerWeights, testingStream.getInstanceRandom());
//                    		testing_Outliner[i] = outlinerInsts_array[index];
                    		testing_Outliner[i] = this.addNoise(outlinerInsts_array[index], testingStream.getInstanceRandom());
                    	}
                    	
                    	testingStream.pauseProgress();
                    	String s = testSetFileCLI;
                    	if (s.substring(s.length()-5, s.length()).equals(".arff")) {
            				s = s.substring(0, s.length()-5);
            			}
            			s = s.concat("_"+(++testSetCounter)+".arff");
            			this.testingARFFFileOption.setValueViaCLIString(s);
            			
            			File testSetFile = this.testingARFFFileOption.getFile();
            			int[] testingInstWrittenByClass = new int[] {0, 0};
            			if (testSetFile != null) {
            				try {
            					Writer testingInstWriter = new BufferedWriter(new FileWriter(testSetFile));
                				monitor.setCurrentActivityDescription("Writing Test Set to ARFF: " + this.testingARFFFileOption.getValueAsCLIString());
                				if (!this.suppressHeaderOption.isSet()){
                					testingInstWriter.write(testingStream.getHeader().toString());
                					testingInstWriter.write("\n");
            					}
                				int testingInstWritten = 0;
                				// Write Borderline
                				for (int i = 0; i < testing_Borderline.length; ++i) {
                					testingInstWriter.write(testing_Borderline[i].toString());
                					testingInstWriter.write("\n");
                					testingInstWrittenByClass[1]++;
                					testingInstWritten++;
                				}
                				// Write Rare
                				for (int i = 0; i < testing_Rare.length; ++i) {
                					testingInstWriter.write(testing_Rare[i].toString());
                					testingInstWriter.write("\n");
                					testingInstWrittenByClass[1]++;
                					testingInstWritten++;
                				}
                				// Write Outliner
                				for (int i = 0; i < testing_Outliner.length; ++i) {
                					testingInstWriter.write(testing_Outliner[i].toString());
                					testingInstWriter.write("\n");
                					testingInstWrittenByClass[1]++;
                					testingInstWritten++;
                				}
                				
//                				System.out.println("Set number: " + testSetCounter);
//                				System.out.println("After writing borderline rare outliner | testingInstWritten: " + testingInstWritten);
//                				System.out.println("=============================================================");
                				while ((testingInstWritten < this.testSetSizeOption.getValue())
                						&& testingStream.hasMoreInstances()) {
                					// Write test set to ARFF
                					Instance to_write = testingStream.nextInstance().getData();
                					if (testingInstWrittenByClass[(int) to_write.classValue()] < (this.testSetSizeOption.getValue() / 2)) {
                						testingInstWriter.write(to_write.toString());
                    					testingInstWriter.write("\n");
                    					testingInstWrittenByClass[(int) to_write.classValue()]++;
                    					testingInstWritten++;
                					}
                					
                				}
                				testingInstWriter.close();
            				} catch (Exception ex) {
            					 throw new RuntimeException(
            	                         "Failed writing to file " + testSetFile, ex);
            				}
            			}
            			testingStream.resumeProgress();
                    }
                }
                trainingInstWriter.close();
                this.testingARFFFileOption.setValueViaCLIString(testSetFileCLI);
    		} catch (Exception ex) {
    			 throw new RuntimeException(
                         "Failed writing to file " + trainingStreamFile, ex);
    		}
    		return "Training stream written to ARFF file " + trainingStreamFile + " | " + testSetCounter + " test sets written to ARFF file.";
    	}
    	throw new IllegalArgumentException("No destination file to write to.");
    	
    }
    
    private Instance addNoise(Instance ori_inst, Random random) {
    	Instance ori = ori_inst.copy();
    	ori.deleteAttributeAt(ori.classIndex());
    	double[] ori_no_class = ori.toDoubleArray();
    	double[] sample = new double[ori_no_class.length+1];
    	for (int i = 0; i < ori_no_class.length; ++i) {
    		sample[i] = ori_no_class[i] + random.nextDouble() * 0.02; // Based on method nextRarePoint() (line 400) in ImbalancedGenerator.java
    	}
    	sample[sample.length-1] = ori_inst.classValue();
		
		Instance to_return = new DenseInstance(1d, sample);
		to_return.setDataset(ori_inst.dataset());
		return to_return;
    }
    
//    private void printStreamsState(ImbalancedDriftGenerator trainingStream, ImbalancedDriftGenerator testingStream) {
//    	System.out.println("Same? " + (trainingStream == testingStream));
//    	System.out.println("=====");
//    	System.out.println("Training Stream MS: " + trainingStream.getModelSeed());
//    	System.out.println("Test set MS: " + testingStream.getModelSeed());
//    	System.out.println("=====");
//    	System.out.println("Training Stream IS: " + trainingStream.getInstanceSeed());
//    	System.out.println("Test set IS: " + testingStream.getInstanceSeed());
//    	System.out.println("=====");
//    	System.out.println("Training Stream Positive Share: " + trainingStream.positiveShareOption.getValue());
//    	System.out.println("Test set Positive Share: " + testingStream.positiveShareOption.getValue());
//    	System.out.println("=====");
//    }

    @Override
    public Class<?> getTaskResultType() {
        return String.class;
    }
}
