package moa.tasks;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.Option;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;

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

public class EvaluatePeriodicHeldOutTestARFF extends ClassificationMainTask {
	
	@Override
    public String getPurposeString() {
        return "Evaluates a classifier on a stream by periodically testing on a heldout set in ARFF file.";
    }
	
	private static final long serialVersionUID = 1L;
	
	public ClassOption learnerOption = new ClassOption("learner", 'l',
            "Classifier to train.", MultiClassClassifier.class, "moa.classifiers.trees.HoeffdingTree");
	
	public ClassOption trainingStreamOption = new ClassOption("trainingStream", 's',
            "Stream to learn from.", ExampleStream.class,
            "ArffFileStream");
	
	public ClassOption testSetOption = new ClassOption("testSet", 'v',
            "Base name of the test sets.", ExampleStream.class,
            "ArffFileStream");
	
	public ClassOption evaluatorOption = new ClassOption("evaluator", 'e',
            "Classification performance evaluation method.",
            LearningPerformanceEvaluator.class,
            "BasicClassificationPerformanceEvaluator");

    public IntOption trainingStreamSizeOption = new IntOption("trainingStreamSize", 'i',
            "Number of training examples, <1 = unlimited.",
            200000, -1, Integer.MAX_VALUE);
    
    public IntOption testSetSizeOption = new IntOption("testSetSize", 'x',
    		"Number of testing examples in each test set.", 500, 0, Integer.MAX_VALUE);
    
    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of training seconds (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency", 'f',
            "How many instances between samples of the learning performance.",
            500, 0, Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public FlagOption cacheTestOption = new FlagOption("cacheTest", 'c',
            "Cache test instances in memory.");
    
    public ClassOption uniformDistributionSetOption = new ClassOption("uniformDistributionSet", 'u',
            "A uniform distribution data set for plotting decision boundary", ExampleStream.class,
            "ArffFileStream");
    
    public StringOption projetDecisionAreasStepsOption = new StringOption("projectionSteps", 'p',
    		"Time steps to project the decision areas \n" + 
    		"The values should be integers and are seperated by semi-colons (;). e.g. \"50;100;150\"\n" +
			"Also, it has to be a multiple of sample Frequency.",
    		"70000;100000");
    
    public FileOption outputPredictionFileOption = new FileOption("outputPredictionFile", 'o',
            "File to append output predictions to.", null, "csv", true);

	@Override
	protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {
		ExampleStream trainingStream = (ExampleStream) getPreparedClassOption(this.trainingStreamOption);
		LearningPerformanceEvaluator evaluator = (LearningPerformanceEvaluator) getPreparedClassOption(this.evaluatorOption);
		LearningCurve learningCurve = new LearningCurve("evaluation instances");
		
		ExampleStream uniformDist_set = (ExampleStream) getPreparedClassOption(this.uniformDistributionSetOption);
		
		int[] projectionSteps = Arrays.stream(this.projetDecisionAreasStepsOption.getValue().split(";")).mapToInt(Integer::parseInt).toArray();
		
		for (int step : projectionSteps) {
			if (step % this.sampleFrequencyOption.getValue() != 0) {
				throw new RuntimeException(
                        "One of the projection steps is not a multiple of sampling frequency.");
			}
		}
		
		// File for dumpping results.
		File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
//                    immediateResultStream = new PrintStream(
//                            new FileOutputStream(dumpFile, true), true);
                	immediateResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(dumpFile, true), 1024*1024), true);
                } else {
//                    immediateResultStream = new PrintStream(
//                            new FileOutputStream(dumpFile), true);
                	immediateResultStream = new PrintStream(new BufferedOutputStream(
                            new FileOutputStream(dumpFile), 1024*1024), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }
        
        boolean firstDump = true;
        int number_of_test_sets = this.trainingStreamSizeOption.getValue() / this.sampleFrequencyOption.getValue();
        ExampleStream[] testSets = new ExampleStream[number_of_test_sets];
        String testSetFileCLI = this.testSetOption.getValueAsCLIString();
        
        Option[] testSetOptions = new Option[number_of_test_sets];
        for (int i = 0; i < testSetOptions.length; ++i) {
        	String tmp_option_name = "testSet" + (i+1);
        	char dummy_char = (char) (i+1000);
        	testSetOptions[i] = new ClassOption(tmp_option_name, dummy_char, "Name of test set.", ExampleStream.class, "ArffFileStream");
        	String s = testSetFileCLI;
        	if (i > 0) {
            	if (s.substring(s.length()-5, s.length()).equals(".arff")) {
    				s = s.substring(0, s.length()-7); // remove "_1.arff"
    			}
            	s = s.concat("_"+(i+1)+".arff");
        	}
        	testSetOptions[i].setValueViaCLIString(s);
        }
        this.config.prepareAdditionalClassOptions(testSetOptions);
        
        for (int i = 0; i < testSets.length; ++i) {			
			if (this.cacheTestOption.isSet()) {
				monitor.setCurrentActivity("Caching test examples...", -1.0);
				ExampleStream tmpTestSetStream = (ExampleStream) getPreparedClassOption((ClassOption) testSetOptions[i]);
				Instances tmptestSet = new Instances(tmpTestSetStream.getHeader(), this.testSetSizeOption.getValue());
				while (tmptestSet.numInstances() < this.testSetSizeOption.getValue()) {
					tmptestSet.add((Instance) tmpTestSetStream.nextInstance().getData());
					if (tmptestSet.numInstances() % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
	                    if (monitor.taskShouldAbort()) {
	                        return null;
	                    }
	                    monitor.setCurrentActivityFractionComplete((double) tmptestSet.numInstances()
	                            / (double) (this.testSetSizeOption.getValue()));
	                }
				}
				testSets[i] = new CachedInstancesStream(tmptestSet);
			} else {
				testSets[i] = (ExampleStream) getPreparedClassOption(this.testSetOption);
			}
        }
        
//        if (this.cacheTestOption.isSet()) {
//        	for (int i = 0; i < testSets.length; ++i) {
//        		System.out.println("Test set " + (i+1) + " size: " + testSets[i].estimatedRemainingInstances());
//        		System.out.println("First instance: ");
//        		System.out.println(testSets[i].nextInstance().getData().toString());
//        	}
//        }
        
        Learner learner = (Learner) getPreparedClassOption(this.learnerOption);
		learner.setModelContext(trainingStream.getHeader());
        
		long instancesProcessed = 0;
		TimingUtils.enablePreciseTiming();
        double totalTrainTime = 0.0;
        int testSetIndex = 0;
        
        int projectionStep_index = 0;
        
        while ((this.trainingStreamSizeOption.getValue() < 1
                || instancesProcessed < this.trainingStreamSizeOption.getValue())
                && trainingStream.hasMoreInstances()) {
        	
        	monitor.setCurrentActivityDescription("Training...");
        	long instancesTarget = instancesProcessed + this.sampleFrequencyOption.getValue();
        	long trainStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        	
        	while (instancesProcessed < instancesTarget && trainingStream.hasMoreInstances()) {
        		learner.trainOnInstance(trainingStream.nextInstance());
        		instancesProcessed++;
        		 if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                     if (monitor.taskShouldAbort()) {
                         return null;
                     }
                     monitor.setCurrentActivityFractionComplete((double) (instancesProcessed)
                             / (double) (this.trainingStreamSizeOption.getValue()));
                 }
        	}
        	double lastTrainTime = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - trainStartTime);
        	totalTrainTime += lastTrainTime;
        	if (this.timeLimitOption.getValue() > 0 && totalTrainTime > this.timeLimitOption.getValue()) {
                break;
            }
        	
        	int projectionStep = projectionSteps[projectionStep_index];
        	
        	if (this.outputPredictionFileOption.getValue() != null && 
    				(instancesProcessed == projectionStep || 
    				instancesProcessed == this.trainingStreamSizeOption.getValue())) {

    			// Set up the corresponding output prediction file.
    			String original_s = this.outputPredictionFileOption.getValueAsCLIString();
    			String s = original_s;
    			
    			if (s.substring(s.length()-4, s.length()).equals(".csv")) {
    				s = s.substring(0, s.length()-4); // remove ".csv"
    			}
            	s = s.concat("_" + (instancesProcessed / 1000) + "k.csv");
    			
            	this.outputPredictionFileOption.setValueViaCLIString(s);
            	
            	File outputPredictionFile = this.outputPredictionFileOption.getFile();
                PrintStream outputPredictionResultStream = null;
                if (outputPredictionFile != null) {
                    try {
                    	if (outputPredictionFile.exists()) {
//                    		outputPredictionResultStream = new PrintStream(new FileOutputStream(outputPredictionFile, true), true);
                            outputPredictionResultStream = new PrintStream(new BufferedOutputStream(
                                    new FileOutputStream(outputPredictionFile, false), 1024*1024), true);
                        } else {
//                        	outputPredictionResultStream = new PrintStream(new FileOutputStream(outputPredictionFile), true);
                            outputPredictionResultStream = new PrintStream(new BufferedOutputStream(
                                    new FileOutputStream(outputPredictionFile), 1024*1024), true);
                        }
                    } catch (Exception ex) {
                        throw new RuntimeException(
                                "Unable to open prediction result file: " + outputPredictionFile, ex);
                    }
                }
                
                uniformDist_set.restart();
                while(uniformDist_set.hasMoreInstances()) {
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
                }
                if (outputPredictionResultStream != null) {
        			outputPredictionResultStream.close();
                }
            	
            	// Reset the output prediction file name.
                this.outputPredictionFileOption.setValueViaCLIString(original_s);
                projectionStep_index = projectionStep_index + 1 >= projectionSteps.length ? projectionStep_index : projectionStep_index + 1;
    		}
        	
        	// Restart the test set and the evaluator.
//        	System.out.println("Test Set now: " + testSetIndex);
        	testSets[testSetIndex].restart();
        	evaluator.reset();
        	
        	
        	
        	long testInstancesProcessed = 0;
        	monitor.setCurrentActivityDescription("Testing (after "
                    + StringUtils.doubleToString(
                    ((double) (instancesProcessed)
                    / (double) (this.trainingStreamSizeOption.getValue()) * 100.0), 2)
                    + "% training)...");
        	long testStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        	int instCount = 0;
        	for (instCount = 0; instCount < this.testSetSizeOption.getValue(); instCount++) {
        		if (testSets[testSetIndex].hasMoreInstances() == false) { // <--- mc: should be checking if "testStream" has more instances instead of "stream".
					break;
				}
        		Example testInst = (Example) testSets[testSetIndex].nextInstance();
        		
//        		if (instCount == 0) {
//                	System.out.println("First instance: \n" + testInst.getData().toString());
//        		} else if (instCount == this.testSetSizeOption.getValue() - 1) {
//        			System.out.println("Last instance: \n" + testInst.getData().toString());
//        		}
        		
        		double trueClass = ((Instance) testInst.getData()).classValue();
        		//testInst.setClassMissing();
                double[] prediction = learner.getVotesForInstance(testInst);
                //testInst.setClassValue(trueClass);
                evaluator.addResult(testInst, prediction);
                testInstancesProcessed++;
                if (testInstancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                    if (monitor.taskShouldAbort()) {
                        return null;
                    }
                    monitor.setCurrentActivityFractionComplete((double) testInstancesProcessed
                            / (double) (this.testSetSizeOption.getValue()));
                }
        	}
        	if (instCount != this.testSetSizeOption.getValue()) {
				break;
			}
        	testSetIndex++;
        	
            double testTime = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - testStartTime);
            
            // Prepare measurements
            List<Measurement> measurements = new LinkedList<Measurement>();
            measurements.add(new Measurement("evaluation instances", instancesProcessed));
            measurements.add(new Measurement("total train time", totalTrainTime));
            measurements.add(new Measurement("total train speed", instancesProcessed / totalTrainTime));
            measurements.add(new Measurement("last train time", lastTrainTime));
            measurements.add(new Measurement("last train speed", this.sampleFrequencyOption.getValue() / lastTrainTime));
            measurements.add(new Measurement("test time", testTime));
            measurements.add(new Measurement("test speed", this.testSetSizeOption.getValue() / testTime));
            
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
        
//        this.testSetOption.setValueViaCLIString(testSetFileCLI);
		
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        return learningCurve;
	}
	
	@Override
	public Class<?> getTaskResultType() {
		return LearningCurve.class;
	}
	
}