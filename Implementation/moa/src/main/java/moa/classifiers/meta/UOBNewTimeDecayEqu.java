/**
 * UOBNewTimeDecayEqu.java
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
 * UOB.java
 * 
 * Author: Leandro L. Minku (l.l.minku@bham.ac.uk)
 * Implementation of Undersampling Online Bagging as in
 * WANG, S.; MINKU, L.L.; YAO, X. "Dealing with Multiple Classes in Online Class Imbalance Learning", 
 * Proceedings of the 25th International Joint Conference on Artificial Intelligence (IJCAI'16), July 2016
 * 
 * Please note that this was not the implementation used in the experiments done in the paper above.
 * It has been implemented after that paper was published.
 * However, it implements the algorithm proposed in that paper. So, it should reflect those results.
 * 
 */

package moa.classifiers.meta;

import com.yahoo.labs.samoa.instances.Instance;

public class UOBNewTimeDecayEqu extends OOBNewTimeDecayEqu {

	@Override
	public String getPurposeString() {
		return "Undersampling on-line bagging of Wang et al IJCAI 2016.";
	}
	
	public UOBNewTimeDecayEqu() {
		super();
	}
	
	// classInstance is the class corresponding to the instance for which we want to calculate lambda
	// will result in an error if classSize is not initialised yet
	@Override
	public double calculatePoissonLambda(Instance inst) {
		double lambda = 1d;
		int minClass = getMinorityClass();
		
		lambda = this.getClassSize(minClass) / this.getClassSize((int) inst.classValue());
		
		return lambda;
	}
}
