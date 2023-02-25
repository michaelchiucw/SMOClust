package moa.tasks;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;

import moa.classifiers.MultiClassClassifier;
import moa.core.Example;
import moa.core.InstanceExample;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.ObjectRepository;
import moa.core.StringUtils;
import moa.core.TimingUtils;
import moa.evaluation.LearningEvaluation;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.learners.Learner;
import moa.options.ClassOption;
import moa.streams.CachedInstancesStream;
import moa.streams.ExampleStream;

public class EvaluateImbalancedLearnWithPeriodicBalancedHeldOutTest extends ClassificationMainTask {
	
    @Override
    public String getPurposeString() {
        return "Evaluates a classifier on a class-imbalanced stream by periodically testing on a class-balanced heldout set.\n"
        		+ "The training stream and the test set MUST follow the same concept but generated with different random seed.";
    }
	
    private static final long serialVersionUID = 1L;
    
    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',
            "Random seed for calculating whether a given example should come from the old or the new concept during the transitional period of a gradual drift", 1);
    
    public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Classifier to train.", MultiClassClassifier.class, "moa.classifiers.trees.HoeffdingTree");
    
    public ClassOption trainingStreamOption = new ClassOption("trainingStream", 's',
            "A class-imbalanced data stream to learn from.", ExampleStream.class,
            "ArffFileStream");
    
    public ClassOption testingExamplesOption = new ClassOption("testingExamples", 'h',
            "A class-balanced data set to test on.", ExampleStream.class,
            "ArffFileStream");
    
    public ClassOption uniformDistributionSetOption = new ClassOption("uniformDistributionSet", 'u',
            "A uniform distribution data set for plotting decision boundary", ExampleStream.class,
            "ArffFileStream");

    public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Learning performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");
    
    public StringOption driftPointsOption = new StringOption("conceptDriftPoints", 'p',
    		"Determine the concept drift points in the data stream.\n" + 
    		"The values should be integers and are seperated by semi-colons (;). e.g. \"1100;400;1500\"\n" + 
    		"Entering the default value of \"-1\" indicates there is NO concept drift in the training stream.",
    		"-1");
    
    public StringOption driftWidthsOption = new StringOption("driftWidths", 'w',
    		"Determine the width of each concept drift in the data stream.\n" + 
    		"The values should be integers and are seperated by semi-colons (;). e.g. \"50;100;150\"\n" + 
    		"Entering the default value of \"-1\" indicates there is NO concept drift in the training stream.",
    		"-1");

    public IntOption testSizeOption = new IntOption("testSize", 'n',
            "Number of testing examples for each concept.\n" +
            "This value is used to slipt the examples in the testingExamples.arff evenly into several test sets" +
            ", corresponding to each concept in the training stream." + 
            "\n\nNote that, this should match the total number of examples in the testingExamples.arff when the driftPointsOption has the value of \"-1\" (indicating that there is no concept drift) " +
            "and will use all the examples in the testingExamples.arff as the test set.",
            500, 0, Integer.MAX_VALUE);

    public IntOption trainSizeOption = new IntOption("trainSize", 'i',
            "Number of training examples, <1 = unlimited.", 0, 0,
            Integer.MAX_VALUE);

    public IntOption trainTimeOption = new IntOption("trainTime", 't',
            "Number of training seconds, <0 = train until there is no more examples in the training stream", -1, Integer.MIN_VALUE, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption(
            "sampleFrequency",
            'f',
            "Number of training examples between samples of learning performance.",
            10, 0, Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);
    
    public FileOption outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", null, "csv", true);
    
	@Override
	protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
		
		Random random = new Random(this.randomSeedOption.getValue());
		
		Learner learner = (Learner) getPreparedClassOption(this.learnerOption);
		ExampleStream testingExamples = (ExampleStream) getPreparedClassOption(this.testingExamplesOption);
		ExampleStream trainingStream = (ExampleStream) getPreparedClassOption(this.trainingStreamOption);
		ExampleStream uniformDist_set = (ExampleStream) getPreparedClassOption(this.uniformDistributionSetOption);
		// Assert the training stream and the test sets have the same header.
		assertTrainingTestingHeaderEqual(testingExamples, trainingStream);
		
		LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
		
		LearningCurve learningCurve = new LearningCurve("evaluation instances");
        
        learner.setModelContext(trainingStream.getHeader());
        
        //File for dumpping results.
        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
        	try {
                if (dumpFile.exists()) {
//                    immediateResultStream = new PrintStream(new FileOutputStream(dumpFile, true), true);
                	immediateResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(dumpFile, false), 1024*1024), true);
                } else {
//                    immediateResultStream = new PrintStream(new FileOutputStream(dumpFile), true);
                	immediateResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(dumpFile), 1024*1024), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        
        //File for output predictions
        File outputPredictionFile = this.outputPredictionFileOption.getFile();
        PrintStream outputPredictionResultStream = null;
        if (outputPredictionFile != null) {
            try {
            	if (outputPredictionFile.exists()) {
//            		outputPredictionResultStream = new PrintStream(new FileOutputStream(outputPredictionFile, true), true);
                    outputPredictionResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(outputPredictionFile, false), 1024*1024), true);
                } else {
//                	outputPredictionResultStream = new PrintStream(new FileOutputStream(outputPredictionFile), true);
                    outputPredictionResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(outputPredictionFile), 1024*1024), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open prediction result file: " + outputPredictionFile, ex);
            }
        }
        
        boolean firstDump = true;
        TimingUtils.enablePreciseTiming();
        
        int testSetSize = this.testSizeOption.getValue();
        
        long instancesProcessed = 0;
        double totalTrainTime = 0.0;
        int concept_index = -1;
        int drift_index = -1;
        
        // Caching the drift points and drift widths
        int[] driftPoints = Arrays.stream(this.driftPointsOption.getValue().split(";")).mapToInt(Integer::parseInt).toArray();
        int[] driftWidths = Arrays.stream(this.driftWidthsOption.getValue().split(";")).mapToInt(Integer::parseInt).toArray();
        // Assert driftPoints and driftWidths params.
        this.assertDriftPointsAndWidthParams(driftPoints, driftWidths);
//        System.out.println("driftPoints: " + Arrays.toString(driftPoints));
//        System.out.println("driftWidths: " + Arrays.toString(driftWidths));
        
        // Check if there is any drift.
		boolean hasDrift = driftPoints.length != 1 || driftPoints[0] != -1 && driftWidths.length != 1 || driftWidths[0] != -1;
//		System.out.println("hasDrift: " + hasDrift);
		
		// Caching all the examples from the testingExample.arff file.
		monitor.setCurrentActivity("Caching the test set...", -1.0);
		CachedInstancesStream[] testSets = null;
		if (hasDrift) {
			testSets = new CachedInstancesStream[driftPoints.length + 1];
			for (int i = 0; i < testSets.length; ++i) {
				monitor.setCurrentActivity("Caching the " + (i+1) + "-th test set...", -1.0);
				testSets[i] = this.cacheTestSet(monitor, testingExamples, (double) testSetSize);
			}
		} else {
			testSets = new CachedInstancesStream[1];
			monitor.setCurrentActivity("Caching the 1st test set...", -1.0);
			testSets[0] = this.cacheTestSet(monitor, testingExamples, (double) testSetSize);
		}
		
		/**
		 * Main loop
		 */
		while ((this.trainSizeOption.getValue() < 1 || instancesProcessed < this.trainSizeOption.getValue()) && trainingStream.hasMoreInstances()) {
			/**
			 * While there are more training examples and not exceeding the limit.
			 */
			monitor.setCurrentActivity("Training...", -1.0);
			long instancesTarget = instancesProcessed + this.sampleFrequencyOption.getValue();
			long trainStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
			/**
			 * Training loop
			 */
			while (instancesProcessed < instancesTarget && trainingStream.hasMoreInstances()) {
				// Train the learner.
				learner.trainOnInstance(trainingStream.nextInstance());
				instancesProcessed++;
				
				// Update the activity fraction.
				if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) (instancesProcessed) / (double) (this.trainSizeOption.getValue()));
                }
			} // End training loop
			
			double lastTrainTime = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - trainStartTime);
            totalTrainTime += lastTrainTime;
            if (this.trainTimeOption.getValue() >= 0 && totalTrainTime > this.trainTimeOption.getValue()) {
                break;
            }
            
            // Decide which test set to be used, based on the number of processed instances.
            if (hasDrift) {
            	int[] result = this.calculateConceptAndDriftIndex(driftPoints, driftWidths, instancesProcessed);
            	concept_index = result[0];
            	drift_index = result[1];
            } else {
            	concept_index = 0;
            	drift_index = -1;
            }
            
            // Restart the test set and the evaluator.
            for (int i = 0; i < testSets.length; ++i) {
            	// If no drift, testSets.length will always be 1 AND concept_index will always be 0.
            	testSets[i].restart();
            }
			evaluator.reset();
			
			long testInstancesProcessed = 0;
			monitor.setCurrentActivityDescription("Testing (after " + 
					StringUtils.doubleToString(((double) (instancesProcessed) / (double) (this.trainSizeOption.getValue()) * 100.0), 2) +
					"% training)...");
			long testStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            int instCount = 0 ;
            /**
             * Testing loop
             */
//          System.out.println("instancesProcessed: " + instancesProcessed);
//    		System.out.println("concept_index: " + concept_index);
            for (instCount = 0; instCount < testSetSize; instCount++) {
            	
            	Example testInst = null;
            	if (drift_index == -1) {
            		// In grace period.
            		if (!testSets[concept_index].hasMoreInstances()) {
            			break;
            		}
            		testInst = testSets[concept_index].nextInstance();
            	} else {
            		// In transitional period.
            		int oldConcept_index = drift_index;
        			int newConcept_index = drift_index + 1;
        			
        			if (!testSets[oldConcept_index].hasMoreInstances() || !testSets[newConcept_index].hasMoreInstances()) {
            			break;
            		}
        			/**
        			 *  TODO: As this sigmoid function is started to be used since the transitional period starts,
        			 *  the sigmoid function is kind of being "trimmed" by the transitional period. i.e. It is not
        			 *  ranging from 0 to 1 but ranging from somewhere larger than 0 to somewhere smaller than 1, depending on the width.
        			 *  N.B.: Dumping the variable "probabilityDrift" while running dumping the variable "probabilityDrift" in
        			 *  ConceptDriftStream.java to see the difference. 
        			 */
        			double x = -4.0 * (double) (instancesProcessed - driftPoints[drift_index]) / (double) driftWidths[drift_index];
        	        double probabilityDrift = 1.0 / (1.0 + Math.exp(x));        	        	
        	        if (random.nextDouble() > probabilityDrift) {
        	        	testInst = testSets[oldConcept_index].nextInstance();
        	        } else {
        	        	testInst = testSets[newConcept_index].nextInstance();
        	        }
            	}
                double trueClass = ((Instance) testInst.getData()).classValue();
                
                double[] prediction = learner.getVotesForInstance(testInst);
                
                evaluator.addResult(testInst, prediction);
                testInstancesProcessed++;
                if (testInstancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) testInstancesProcessed
                            / (double) (testSetSize));
                }
            }// End testing loop
            
            if (instCount != testSetSize) {
				break;
			}
            double testTime = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - testStartTime);
            
            // Prepare measurements
            List<Measurement> measurements = new LinkedList<Measurement>();
            measurements.add(new Measurement("evaluation instances", instancesProcessed));
            measurements.add(new Measurement("total train time", totalTrainTime));
            measurements.add(new Measurement("total train speed", instancesProcessed / totalTrainTime));
            measurements.add(new Measurement("last train time", lastTrainTime));
            measurements.add(new Measurement("last train speed", this.sampleFrequencyOption.getValue() / lastTrainTime));
            measurements.add(new Measurement("test time", testTime));
            measurements.add(new Measurement("test speed", this.testSizeOption.getValue() / testTime));
            
            Measurement[] performanceMeasurements = evaluator.getPerformanceMeasurements();
            for (Measurement measurement : performanceMeasurements) {
                measurements.add(measurement);
            }
            
            Measurement[] modelMeasurements = learner.getModelMeasurements();
            for (Measurement measurement : modelMeasurements) {
                measurements.add(measurement);
            }
            
            // Insert measurements to learning curve
            learningCurve.insertEntry(new LearningEvaluation(measurements.toArray(new Measurement[measurements.size()])));
            
            if (immediateResultStream != null) {
                if (firstDump) {
                    immediateResultStream.println(learningCurve.headerToString());
                    firstDump = false;
                }
                immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                immediateResultStream.flush();
            }
            if (monitor.resultPreviewRequested()) {
                monitor.setLatestResultPreview(learningCurve.copy());
            }
		}// End while
		
		while (this.outputPredictionFileOption.getValue() != null && uniformDist_set.hasMoreInstances()) {
			InstanceExample example = (InstanceExample) uniformDist_set.nextInstance();
			double[] prediction = learner.getVotesForInstance(example);
			double predicted_class = MiscUtils.maxIndex(prediction);
			
			Instance inst = example.getData();
			
			StringBuilder to_print_builder = new StringBuilder();
			for (int i = 0; i < inst.numAttributes(); ++i) {
				if (inst.attribute(i) != inst.classAttribute()) {
					to_print_builder.append(inst.value(i));
				} else {
					to_print_builder.append(predicted_class);
				}
				if (i < inst.numAttributes() - 1) {
					to_print_builder.append(",");
				}
			}
			outputPredictionResultStream.println(to_print_builder.toString());
			outputPredictionResultStream.flush();
		}// End while
		
		if (immediateResultStream != null) {
            immediateResultStream.close();
        }
		if (outputPredictionResultStream != null) {
			outputPredictionResultStream.close();
        }
        
		return learningCurve;
	}
	
	/**
	 * @param driftPoints
	 * @param driftWidths
	 * @param instancesProcessed
	 * @return int[0] = concept index; int[1] = drift index
	 * 		   If drift index is -1, it means we are currently at grace period.
	 * 		   Otherwise, the value at int[1] refers to the transitional period of which drift we are currently in.
	 */
	private int[] calculateConceptAndDriftIndex(int[] driftPoints, int[] driftWidths, long instancesProcessed) {
		int[] to_return = new int[2];
		
		// Determine which concept we are currently in.
		int concept_index = 0;
		if (driftPoints.length != 1 || driftPoints[0] != -1) {
			for (int driftPoint : driftPoints) {
				if (instancesProcessed >= driftPoint) {
					concept_index++;
				} else {
					break;
				}
			}
		}
		to_return[0] = concept_index;
		
		// Determine whether we are in a concept grace period or in a transitional period.
		if (concept_index == 0) {
			// At the first concept
//			System.out.println("instancesProcessed: " + instancesProcessed + " | calculateConceptAndDriftIndex | At the first concept | concept_index: " + concept_index);
			int half_width = driftWidths[0] / 2;
			
			to_return[1] = driftPoints[0] - half_width <= instancesProcessed ? 0 : -1;
		} else if (concept_index == driftPoints.length) {
			// At the last concept
//			System.out.println("instancesProcessed: " + instancesProcessed + " | calculateConceptAndDriftIndex | At the last concept | concept_index: " + concept_index);
			int half_width = driftWidths[driftWidths.length - 1] /2;
			
			to_return[1] = instancesProcessed <= driftPoints[driftPoints.length - 1] + half_width ? (driftWidths.length - 1) : -1;
		} else {
			// At other concepts.
			// head drift -> current concept -> tail drift
//			System.out.println("instancesProcessed: " + instancesProcessed + " | calculateConceptAndDriftIndex | At the other concept | concept_index: " + concept_index);
			int drift_point_head = driftPoints[concept_index - 1];
			int drift_point_tail = driftPoints[concept_index];
			
			int half_width_head = driftWidths[concept_index - 1] / 2;
			int half_width_tail = driftWidths[concept_index] / 2;
			
			if (instancesProcessed <= drift_point_head + half_width_head) {
				to_return[1] = concept_index - 1;
			} else if (drift_point_tail - half_width_tail <= instancesProcessed) {
				to_return[1] = concept_index;
			} else {
				to_return[1] = -1;
			}
		}
		
//		System.out.println("to_return: " + Arrays.toString(to_return));
		return to_return;
	}
	
	private CachedInstancesStream cacheTestSet(TaskMonitor monitor, ExampleStream testingExamples, double testSetSize) {
		// Initialise the test set holder
		Instances testSet_Instances = new Instances(testingExamples.getHeader(), this.testSizeOption.getValue());
		// Caching testing examples from .arff file to test set.
		while (testSet_Instances.numInstances() < testSetSize) {
			if (testingExamples.hasMoreInstances()) {
				testSet_Instances.add((Instance) testingExamples.nextInstance().getData());
				if (testSet_Instances.numInstances() % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
					if (monitor.taskShouldAbort()) {
						return null;
					}
					monitor.setCurrentActivityFractionComplete((double) testSet_Instances.numInstances() / (double) testSetSize);
				}
			} else {
				throw new IndexOutOfBoundsException("Please make sure the number of example in the testingExample.arff is a multiple of the predefined size of a test set.");
			}
		}
		return new CachedInstancesStream(testSet_Instances);
	}
	
	private void assertTrainingTestingHeaderEqual(ExampleStream stream1, ExampleStream stream2) {
		InstancesHeader stream1_header = stream1.getHeader();
		InstancesHeader stream2_header = stream2.getHeader();
		
		if (stream1_header.numAttributes() != stream2_header.numAttributes()) {
			throw new IllegalArgumentException("Please make sure the numbers of attrubutes of the testing examples and the training stream are the same.");
		}
		if (stream1_header.numInputAttributes() != stream2_header.numInputAttributes()) {
			throw new IllegalArgumentException("Please make sure the numbers of input attrubutes of the testing examples and the training stream are the same.");
		}
		if (stream1_header.numClasses() != stream2_header.numClasses()) {
			throw new IllegalArgumentException("Please make sure the numbers of classes of the testing examples and the training stream are the same.");
		}
		if (stream1_header.numOutputAttributes() != stream2_header.numOutputAttributes()) {
			throw new IllegalArgumentException("Please make sure the numbers of output attrubutes of the testing examples and the training stream are the same.");
		}
		
		for (int i = 0; i < stream1_header.numAttributes(); ++i) {
			Attribute stream1_attribute = stream1_header.attribute(i);
			Attribute stream2_attribute = stream2_header.attribute(i);
			if (!stream1_attribute.name().equals(stream2_attribute.name())) {
				throw new IllegalArgumentException("Please make sure the " + i + "-th attributes are equal:" + 
												   "\nStream1_attribute: " + stream1_attribute.toString() +
												   "\nStream2_attribute: " + stream2_attribute.toString());
			}
			if (stream1_attribute.isNominal() && stream2_attribute.isNominal()) {
				List<String> testingAttr_values = stream1_attribute.getAttributeValues();
				List<String> trainingAttr_values = stream2_attribute.getAttributeValues();
				
				if (!testingAttr_values.equals(trainingAttr_values)) {
					throw new IllegalArgumentException("Please make sure the nominal attribute at index " + i + " has the same list of values.");
				}
				
			} else if (stream1_attribute.isNominal() && stream2_attribute.isNumeric() || 
					stream1_attribute.isNumeric() && stream2_attribute.isNominal()) {
				throw new IllegalArgumentException("Please make sure the " + i + "-th attributes are in the same type.");
			}
		}
	}
	
	private void assertDriftPointsAndWidthParams(int[] driftPoints, int[] driftWidths) {
		// Assert paramters of drift points and drift widths match each other.
		if (driftPoints.length != driftWidths.length) {
        	throw new IllegalArgumentException("Please make sure the number of drifts and the number of drift widths are the same.");
        } else if (driftPoints.length == 1 && driftWidths.length == 1 && 
        		  ((driftPoints[0] == -1 && driftWidths[0] != -1) || (driftPoints[0] != -1 && driftWidths[0] == -1))) {
        	throw new IllegalArgumentException("Please make sure both driftPointsOption and driftWidthsOption indicate there is no concept drift." +
					 						   "\ndriftPoints.length: " + driftPoints.length +
					 						   "\ndriftWidths.length: " + driftWidths.length +
					 						   "\ndriftPoints[0]: " + driftPoints[0] +
					 						   "\ndriftWidths[0]: " + driftWidths[0]);
        }
        // Assert the drift width is >= 1 when there is at least one drift.
        if (driftPoints[0] != -1) {
        	for (int width : driftWidths) {
        		if (width < 1) {
        			throw new IllegalArgumentException("Please make sure the width of the drift is >= 1.");
        		}
        	}
        }
	}
	
	@Override
	public Class<?> getTaskResultType() {
		return null;
	}

}
