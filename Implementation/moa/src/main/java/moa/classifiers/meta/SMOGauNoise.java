package moa.classifiers.meta;

import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import com.github.javacliparser.FileOption;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import com.yahoo.labs.samoa.instances.WekaToSamoaInstanceConverter;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.driftdetection.DDM_GMean;
import moa.classifiers.core.driftdetection.DDM_OCI;
import moa.classifiers.core.driftdetection.PMAUC_EWAUC_WAUC_GMean_PH;
import moa.core.Measurement;
import moa.core.Utils;
import moa.options.ClassOption;

public class SMOGauNoise extends AbstractClassifier implements MultiClassClassifier {
	
	private static final long serialVersionUID = 1L;
	
	public IntOption randSeedOption = new IntOption("seed", 'i',
            "Seed for random behaviour of the classifier.", 1);
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "meta.OzaBag");
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	public FloatOption gaussianNoiseVarianceOption = new FloatOption("gaussianNoiseVariance", 'v',
			"Variance for Gaussian noise to create synthetic examples", 0.01, 0.0, Double.MAX_VALUE);
	
	public FloatOption categoricalChangeProbabilityOption = new FloatOption("categoricalChangeProbability", 'c',
			"Probability for categorical attributes to change to another value.", 0.2, 0.0, 1.0);
	
	public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");
	
	public FlagOption disableDDMOption = new FlagOption("disableDDM", 'z', "disableDDM");
	
//	public FileOption exportSyntheticDataFileOption = new FileOption("synthData", 'o',
//			"File to append all the synthetic data generated.", null, "csv", true);
//	
//	// TODO: for development use
//	public FlagOption dumpGauNoiseSynthInstOption = new FlagOption("dumpGauNoiseSynthInst", 'g', "dumpGauNoiseSynthInst");
	
	protected BaseLearner baseLearner;
	
	protected SamoaToWekaInstanceConverter moaToWekaInstanceConverter;
	protected WekaToSamoaInstanceConverter wekaToMoaInstanceConverter;
	
	protected Instance[] last_inst;
	
	protected ChangeDetector driftDetector;
	protected DRIFT_LEVEL drift_level;
	protected double[] timeStepsAfterDrift;
	protected int driftCount;
	
//	protected boolean firstDumpSynth;

    @Override
    public String getPurposeString() {
        return "SMOGauNoise";
    }
	
	@Override
    public void resetLearningImpl() {
		this.randomSeed = this.randSeedOption.getValue();
		this.classifierRandom = new Random(this.randomSeed);
		
		this.baseLearner = new BaseLearner(((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy());
		
		this.moaToWekaInstanceConverter = new SamoaToWekaInstanceConverter();
		this.wekaToMoaInstanceConverter = new WekaToSamoaInstanceConverter();
		
		this.last_inst = null;
		
		this.driftDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectorOption)).copy();
		this.driftCount = 0;
		this.timeStepsAfterDrift = null;
		
//		this.firstDumpSynth = true;
	}
	
	@Override
    public void trainOnInstanceImpl(Instance inst) {
		if (this.timeStepsAfterDrift == null) {
			this.timeStepsAfterDrift = new double[inst.numClasses()];
			for (int i = 0; i < this.timeStepsAfterDrift.length; ++i) {
				this.timeStepsAfterDrift[i] = 0.0;
			}
		}
		
		// Prepare and initialise the clusterers.
		if (this.last_inst == null) {
			last_inst = new Instance[inst.numClasses()];
			for (int i=0; i < last_inst.length; ++i) {
				last_inst[i] = null;
			}
		}
		
		/*
		 * Concept Drift Detection
		 */
		if (!this.disableDDMOption.isSet()) {
			double prediction = Utils.maxIndex(this.getVotesForInstance(inst)) == inst.classValue() ? 0.0 : 1.0;
			/**
			 * DDM_OCI has to put before DDM_GMean because of polymorphism.
			 * DDM_OCI is a subclass of DDM_GMean
			 */
			if (this.driftDetector instanceof DDM_OCI) {
	        	((DDM_OCI) this.driftDetector).input(prediction, inst);
	        } else if (this.driftDetector instanceof DDM_GMean) {
				((DDM_GMean) this.driftDetector).input(prediction, inst);
	        } else if (this.driftDetector instanceof PMAUC_EWAUC_WAUC_GMean_PH) {
	        	((PMAUC_EWAUC_WAUC_GMean_PH) this.driftDetector).input(this.getVotesForInstance(inst), inst);
	        } else {
	        	this.driftDetector.input(prediction);
	        }
			
			this.drift_level = DRIFT_LEVEL.NORMAL;
			if (this.driftDetector.getChange()) {
				this.drift_level = DRIFT_LEVEL.OUTCONTROL;
			}
			
			switch (this.drift_level) {
				case NORMAL:
					this.timeStepsAfterDrift[(int) inst.classValue()]++;
					break;
				case OUTCONTROL:
					this.baseLearner = new BaseLearner(((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy());
					for (int i = 0; i < this.timeStepsAfterDrift.length; ++i) {
						this.timeStepsAfterDrift[i] = (i == (int) inst.classValue()) ? 1.0 : 0.0;
					}
					this.driftCount++;
					break;
				default:
					System.out.print("ERROR!");
					break;
			}
		}
		
		int current_class = (int) inst.classValue();
//		System.out.println("\n\ncurrent_class: " + current_class);
		
		/**
		 * Train ensemble
		 */
//		this.baseLearner.printClassSize("\nBefore training...");
		this.baseLearner.trainOnInstance(inst);
		this.last_inst[current_class] = inst.copy();
		
		int maj_class = this.baseLearner.getMajorityClass();
		int min_class = this.baseLearner.getMinorityClass();

		
//		this.baseLearner.printClassSize("\nBefore Catch up train...");
//		List<Instance> synthInsts = new LinkedList<Instance>();
		while (this.baseLearner.getRawClassSize(min_class) < this.baseLearner.getRawClassSize(maj_class) && this.last_inst[min_class] != null) {
			// Simply Add Noise to the most recent example to create synthtic data (ignoring categorical attributes)
    		Instance synthInst = this.addGaussianNoiseToInstance(this.last_inst[min_class]);
//    		System.out.println("Oringinal: \n" + Arrays.toString(this.last_inst[min_class].toDoubleArray()));
//    		System.out.println("Synth: \n" + Arrays.toString(synthInst.toDoubleArray()));
//    		System.out.println("Distance: " + this.distance(this.last_inst[min_class], synthInst));
//    		synthInsts.add(synthInst);
    		this.baseLearner.trainOnInstance(synthInst);
			
		} // End-while
		
    	// Dumping synthetic instances
//		if (synthInsts.size() > 0) {
////			System.out.println("synthInsts.length: " + synthInsts.size());
//			this.dumpSynthInst(synthInsts, this.firstDumpSynth);
//			firstDumpSynth = false;
//		}
		
//		this.baseLearner.printClassSize("\nAfter Catch up train...");
    }
	
//	//TODO: for debugging
//	private double distance(Instance pt1, Instance pt2) {
//		double distance = 0;
//		for (int i = 0; i < pt1.numInputAttributes(); ++i) {
//			distance += (pt1.value(i) - pt2.value(i)) * (pt1.value(i) - pt2.value(i));
//		}
//		return Math.sqrt(distance);
//	}
	
	// will result in an error if classSize is not initialised yet
	@Override
    protected Measurement[] getModelMeasurementsImpl() {
		return this.baseLearner.getModelMeasurements();
    }

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return true;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return this.baseLearner.getVotesForInstance(inst);
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}
	
	private Instance addGaussianNoiseToInstance(Instance point) {
		Instance pointWithNoise = point.copy();
		double variance = this.gaussianNoiseVarianceOption.getValue();
		int classIndex = pointWithNoise.classIndex();

		for (int i = 0; i < pointWithNoise.numAttributes(); ++i) {
			Attribute currentAttr = pointWithNoise.attribute(i);
			if (i != classIndex && currentAttr.isNumeric()) {
				// Numerical attribute
				double newValue = this.classifierRandom.nextGaussian() * Math.sqrt(variance) + pointWithNoise.value(i);
				pointWithNoise.setValue(i, newValue);
			} else if (i != classIndex && currentAttr.isNominal()) {
				// Nominal attribute
				int numPossibleValues = currentAttr.numValues();
				double currentValue = pointWithNoise.value(i);
				double randN = this.classifierRandom.nextDouble();
				if (randN < this.categoricalChangeProbabilityOption.getValue()) {
					// Change to another value
					double rand_index = -1;
					do {
						rand_index = (double) this.classifierRandom.nextInt(numPossibleValues); // 0 <= randN < allPossibleValues.size()
					} while(rand_index == currentValue);
					pointWithNoise.setValue(i, rand_index);
				} else {
					// retain the same value
					continue;
				}
			}
		}
		
		return pointWithNoise;
	}
	
//	private void dumpSynthInst(List<Instance> insts, boolean firstDump) {
//		//File for dumpping results.
//		File dumpSynthInstFile = this.exportSyntheticDataFileOption.getFile();
//		PrintStream dumpSynthInstStream = null;
//        if (dumpSynthInstFile != null) {
//        	try {
//        		dumpSynthInstStream = new PrintStream(new BufferedOutputStream(
//                        new FileOutputStream(dumpSynthInstFile, !firstDump), 1024*1024), true);
//            } catch (Exception ex) {
//                throw new RuntimeException(
//                        "Unable to open immediate result file: " + dumpSynthInstFile, ex);
//            }
//        	
//        	for (Instance inst : insts) {
//            	dumpSynthInstStream.println(convertInstToCSVformat(inst));
//            	dumpSynthInstStream.flush();
//            }
//        	dumpSynthInstStream.close();
//        }
//	}
	
//	private String convertInstToCSVformat(Instance inst) {
//		StringBuilder to_print_builder = new StringBuilder();
//		for (int i = 0; i < inst.numAttributes(); ++i) {
//			to_print_builder.append(inst.value(i));
//			if (i <  inst.numAttributes() - 1) {
//				to_print_builder.append(",");
//			}
//		}
//		return to_print_builder.toString();
//	}
	
	protected class BaseLearner extends AbstractClassifier {
		
		protected Classifier raw_learner;
		
		protected double[] classSizeEstimation;
		protected double[] b;
		
		public BaseLearner(Classifier raw_learner_type) {
			this.raw_learner = raw_learner_type.copy();
			if (this.raw_learner instanceof WEKAClassifier) {
				((WEKAClassifier) this.raw_learner).buildClassifier();
			}
			this.resetLearning();
		}

		@Override
		public boolean isRandomizable() {
			// TODO Auto-generated method stub
			return false;
		}

		@Override
		public double[] getVotesForInstance(Instance inst) {
			// TODO Auto-generated method stub
			return this.raw_learner.getVotesForInstance(inst);
		}

		@Override
		public void resetLearningImpl() {
			// TODO Auto-generated method stub
			this.raw_learner.resetLearning();
			this.classSizeEstimation = null;
			this.b = null;
		}

		@Override
		public void trainOnInstanceImpl(Instance inst) {
			this.raw_learner.trainOnInstance(inst);
			this.updateClassSize(inst);
		}

		@Override
		protected Measurement[] getModelMeasurementsImpl() {
			Measurement[] measure = null;
			if (classSizeEstimation != null && b != null) {
				measure = new Measurement[classSizeEstimation.length * 2];
				for (int i=0; i<classSizeEstimation.length; ++i) {
					String str = "[Normalised] size of class " + i;
					measure[i] = new Measurement(str,this.getNormalisedClassSize(i));
				}
				for (int i=0; i<classSizeEstimation.length; ++i) {
					String str = "[Raw] size of class " + i;
					measure[i+classSizeEstimation.length] = new Measurement(str,this.getRawClassSize(i));
				}
			}
			return measure;
		}

		@Override
		public void getModelDescription(StringBuilder out, int indent) {
			// TODO Auto-generated method stub
			
		}
		
		public void printClassSize(String msg) {
			if (this.classSizeEstimation == null || this.b == null) {
				System.out.println("Class Size not initialised.");
			} else {
				System.out.println(msg);
				for (int i = 0; i < this.classSizeEstimation.length; ++i) {
					System.out.println("[Normalised] Class Size of class " + i + ": " + this.getNormalisedClassSize(i));
					System.out.println("[Raw] Class Size of class " + i + ": " + this.getRawClassSize(i));
				}
			}
		}
		
		public void updateClassSize(Instance inst) {
			if (this.classSizeEstimation == null) {
				classSizeEstimation = new double[inst.numClasses()];

				// <---start class size as equal for all classes
				for (int i=0; i<classSizeEstimation.length; ++i) {
					classSizeEstimation[i] = 1d/classSizeEstimation.length;
				}
			}
			if (this.b == null) {
				b = new double[inst.numClasses()];
				
				for (int i=0; i<classSizeEstimation.length; ++i) {
					b[i] = 1d/b.length;
				}
			}
			
			for (int i=0; i<classSizeEstimation.length; ++i) {
				classSizeEstimation[i] = thetaOption.getValue() * classSizeEstimation[i] + ((int) inst.classValue() == i ? 1d:0d);
				b[i] = thetaOption.getValue() * b[i] + 1d;
			}
		}
		
		public double getNormalisedClassSize(int classIndex) {
			return b[classIndex] > 0.0 ? classSizeEstimation[classIndex] / b[classIndex] : 0.0;
		}
		
		public double getRawClassSize(int classIndex) {
			return classSizeEstimation[classIndex];
		}
		
		// will result in an error if classSize is not initialised yet
		public int getMajorityClass() {
			int indexMaj = 0;

			for (int i=1; i<classSizeEstimation.length; ++i) {
				if (this.getNormalisedClassSize(i) > this.getNormalisedClassSize(indexMaj)) {
					indexMaj = i;
				}
			}
			return indexMaj;
		}
		
		// will result in an error if classSize is not initialised yet
		public int getMinorityClass() {
			int indexMin = 0;

			for (int i=1; i<classSizeEstimation.length; ++i) {
				if (this.getNormalisedClassSize(i) <= this.getNormalisedClassSize(indexMin)) {
					indexMin = i;
				}
			}
			return indexMin;
		}
		
	}
	
	protected enum DRIFT_LEVEL {
		NORMAL, WARNING, OUTCONTROL
	}
}