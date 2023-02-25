/**
 * OOBNewTimeDecayEqu.java
 * 
 * Author: Chun Wai Chiu (cxc1015@bham.ac.uk)
 * 
 * Update to use the new calculation of time decaying factor:
 * x_t = theta * x_(t-1) + new_value
 * b_t = theta * b_(t-1) + 1.0
 * 
 * getNormalisedValue = x_t / b_t
 * 
 * Based on the following file:
 * 
 * OOB.java
 * 
 * Author: Leandro L. Minku (l.l.minku@bham.ac.uk)
 * Implementation of Oversampling Online Bagging as in
 * WANG, S.; MINKU, L.L.; YAO, X. "Dealing with Multiple Classes in Online Class Imbalance Learning", 
 * Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI'16), July 2016
 * 
 * Please note that this was not the implementation used in the experiments done in the paper above.
 * It has been implemented after that paper was published.
 * However, it implements the algorithm proposed in that paper. So, it should reflect those results.
 * 
 */

package moa.classifiers.meta;

import java.util.Random;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.Measurement;
import moa.core.MiscUtils;

public class OOBNewTimeDecayEqu extends OzaBag {
	
	private static final long serialVersionUID = 1L;
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	public IntOption randSeedOption = new IntOption("seed", 'i',
            "Seed for random behaviour of the classifier.", 1);
	
	protected double[] classSizeEstimation; // time-decayed size of each class
	protected double[] b;

    @Override
    public String getPurposeString() {
        return "Oversampling on-line bagging of Wang et al IJCAI 2016.";
    }
	
	public OOBNewTimeDecayEqu() {
		super();
		classSizeEstimation = null;
		b = null;
	}
	
	@Override
    public void resetLearningImpl() {
		super.resetLearningImpl();
		this.randomSeed = this.randSeedOption.getValue();
		this.classifierRandom = new Random(this.randomSeed);
	}
	
	@Override
    public void trainOnInstanceImpl(Instance inst) {
		updateClassSize(inst);
		double lambda = calculatePoissonLambda(inst);
		
		for (int i = 0; i < this.ensemble.length; i++) {
            int k = MiscUtils.poisson(lambda, this.classifierRandom);
            if (k > 0) {
                Instance weightedInst = (Instance) inst.copy();
                weightedInst.setWeight(inst.weight() * k);
                this.ensemble[i].trainOnInstance(weightedInst);
            }
        }
    }
	
	protected void updateClassSize(Instance inst) {
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
	
	protected double getClassSize(int classIndex) {
		return b[classIndex] > 0.0 ? classSizeEstimation[classIndex] / b[classIndex] : 0.0;
	}
	
	// classInstance is the class corresponding to the instance for which we want to calculate lambda
	// will result in an error if classSize is not initialised yet
	public double calculatePoissonLambda(Instance inst) {
		double lambda = 1d;
		int majClass = getMajorityClass();
		
		lambda = this.getClassSize(majClass) / this.getClassSize((int) inst.classValue());
		
		return lambda;
	}

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
		Measurement [] measure = super.getModelMeasurementsImpl();
		Measurement [] measurePlus = null;
		
		if (classSizeEstimation != null && b != null) {
			measurePlus = new Measurement[measure.length + classSizeEstimation.length];
			for (int i=0; i<measure.length; ++i) {
				measurePlus[i] = measure[i];
			}

			for (int i=0; i<classSizeEstimation.length; ++i) {
				String str = "size of class " + i;
				measurePlus[measure.length+i] = new Measurement(str,this.getClassSize(i));
			}
		} else
			measurePlus = measure;
		
		return measurePlus;
    }
}