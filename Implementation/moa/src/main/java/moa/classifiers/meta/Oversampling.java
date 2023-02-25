package moa.classifiers.meta;

import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;

public class Oversampling extends AbstractClassifier implements MultiClassClassifier {
	
	private static final long serialVersionUID = 1L;
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "meta.OzaBag");
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	public IntOption randSeedOption = new IntOption("seed", 'i',
            "Seed for random behaviour of the classifier.", 1);
	
	protected Classifier baseLearner;
	
	protected double[] classSizeEstimation; // time-decayed class size of each class
	protected double[] b;
	
	protected Instance[] last_inst;
	protected boolean hasInitStatArrays;

    @Override
    public String getPurposeString() {
        return "Oversampling: Reuse the last seen example of the minoirty class for oversampling.";
    }
	
	@Override
    public void resetLearningImpl() {
		this.randomSeed = this.randSeedOption.getValue();
		this.classifierRandom = new Random(this.randomSeed);
		
		this.baseLearner = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		if (baseLearner instanceof WEKAClassifier) {
			((WEKAClassifier) baseLearner).buildClassifier();
		}
		baseLearner.resetLearning();
		
		classSizeEstimation = null;
		b = null;
		last_inst = null;
		
		hasInitStatArrays = false;
	}
	
	@Override
    public void trainOnInstanceImpl(Instance inst) {
		if (!this.hasInitStatArrays) {
			initStatArrays(inst);
		}
		int current_class = (int) inst.classValue();
		
//		System.out.println("\n\ncurrent_class: " + current_class);
//		printClassSize("\nBefore training...");
		this.baseLearner.trainOnInstance(inst);
		this.updateClassSize(inst);
		this.last_inst[current_class] = inst.copy();
		
		int maj_class = this.getMajorityClass();
		int min_class = this.getMinorityClass();
		
//		printClassSize("\nBefore Catch up train...");
		int counter = 0;
		while (this.classSizeEstimation[min_class] < this.classSizeEstimation[maj_class] && 
				this.last_inst[min_class] != null) {
//			double classSizeEst_diff = this.classSizeEstimation[maj_class] - this.classSizeEstimation[min_class];
			this.baseLearner.trainOnInstance(this.last_inst[min_class]);
			this.updateClassSize(this.last_inst[min_class]);
//			printClassSize("\nDuring Catch up train... counter: " + counter++);
		}
//		printClassSize("\nAfter Catch up train...");
    }
	
	protected void printClassSize(String msg) {
		System.out.println(msg);
		for (int i = 0; i < this.classSizeEstimation.length; ++i) {
			System.out.println("Class Size of class " + i + ": " + this.getClassSize(i));
			System.out.println("[Raw] Class Size of class " + i + ": " + this.classSizeEstimation[i]);
		}
	}
	
	protected void initStatArrays(Instance inst) {
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
		if (this.last_inst == null) {
			last_inst = new Instance[inst.numClasses()];
			for (int i=0; i < last_inst.length; ++i) {
				last_inst[i] = null;
			}
		}
		hasInitStatArrays = true;
	}
	
	protected void updateClassSize(Instance inst) {
		for (int i=0; i<classSizeEstimation.length; ++i) {
			classSizeEstimation[i] = thetaOption.getValue() * classSizeEstimation[i] + ((int) inst.classValue() == i ? 1d:0d);
			b[i] = thetaOption.getValue() * b[i] + 1d;
		}
	}
	
	protected double getClassSize(int classIndex) {
		return b[classIndex] > 0.0 ? classSizeEstimation[classIndex] / b[classIndex] : 0.0;
	}
	
	// classInstance is the class corresponding to the instance for which we want to calculate lambda
	// will result in an error if classSize is not initialised yet
//	public double calculatePoissonLambda(Instance inst) {
//		double lambda = 1d;
//		int majClass = getMajorityClass();
//		
//		lambda = this.getClassSize(majClass) / this.getClassSize((int) inst.classValue());
//		
//		return lambda;
//	}

	// will result in an error if classSize is not initialised yet
	public int getMajorityClass() {
		int indexMaj = 0;

		for (int i=1; i<classSizeEstimation.length; ++i) {
			if (this.getClassSize(i) > this.getClassSize(indexMaj)) {
				indexMaj = i;
			}
		}
		return indexMaj;
	}
	
	// will result in an error if classSize is not initialised yet
	public int getMinorityClass() {
		int indexMin = 0;

		for (int i=1; i<classSizeEstimation.length; ++i) {
			if (this.getClassSize(i) <= this.getClassSize(indexMin)) {
				indexMin = i;
			}
		}
		return indexMin;
	}
	
	// will result in an error if classSize is not initialised yet
	@Override
    protected Measurement[] getModelMeasurementsImpl() {
		Measurement[] measure = null;
		if (classSizeEstimation != null && b != null) {
			measure = new Measurement[classSizeEstimation.length * 2];
			for (int i=0; i<classSizeEstimation.length; ++i) {
				String str = "[Normalised] size of class " + i;
				measure[i] = new Measurement(str,this.getClassSize(i));
			}
			for (int i=0; i<classSizeEstimation.length; ++i) {
				String str = "[Raw] size of class " + i;
				measure[i+classSizeEstimation.length] = new Measurement(str,classSizeEstimation[i]);
			}
		}
		return measure;
    }

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
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
}