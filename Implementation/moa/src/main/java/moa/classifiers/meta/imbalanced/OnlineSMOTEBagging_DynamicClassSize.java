/*
 *  OnlineSMOTEBagging.java
 *  
 *  @author Alessio Bernardo (alessio dot bernardo at polimi dot dot it)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
        
 */
package moa.classifiers.meta.imbalanced;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;
import weka.core.Attribute;
import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Random;

import moa.classifiers.core.driftdetection.ADWIN;
import moa.classifiers.lazy.neighboursearch.LinearNNSearch;
import moa.classifiers.lazy.neighboursearch.NearestNeighbourSearch;


/**
 *  Online SMOTEBagging is the online version of the ensemble method SMOTEBagging.
 *
 * <p>This method works by re-sampling the negative class with replacement rate at 100%,
    while :math:`CN^+` positive examples are generated for each base learner, among
    which :math:`a\%` of them are created by re-sampling and the rest of the examples
    are created by the synthetic minority oversampling technique (SMOTE).</p>

    <p>This online ensemble learner method is improved by the addition of an ADWIN change
    detector. ADWIN stands for Adaptive Windowing. It works by keeping updated
    statistics of a variable sized window, so it can detect changes and
    perform cuts in its window to better adapt the learning algorithms.</p>
 *
 * <p>See details in:<br> B. Wang and J. Pineau, "Online Bagging and Boosting for Imbalanced Data Streams,"
       in IEEE Transactions on Knowledge and Data Engineering, vol. 28, no. 12, pp.
       3353-3366, 1 Dec. 2016. doi: 10.1109/TKDE.2016.2609424</p>
 *
 * <p>Parameters:</p> <ul>
 * <li>-l : Each classiﬁer to train of the ensemble is an instance of the base estimator.</li>
 * <li>-s : The size of the ensemble, in other words, how many classifiers to train.</li>
 * <li>-i : The sampling rate of the positive instances.</li>
 * <li>-d : Should use ADWIN as drift detector? If enabled it is used by the method 
 * 	to track the performance of the classifiers and adapt when a drift is detected.</li>
 * <li>-r : Seed for the random state.</li>
 * </ul>
 *
 * @author Alessio Bernardo (alessio dot bernardo at polimi dot dot it)
 * @version $Revision: 1 $
 */
public class OnlineSMOTEBagging_DynamicClassSize extends AbstractClassifier implements MultiClassClassifier {

    @Override
    public String getPurposeString() {
        return "OnlineAdaC2 is the adaptation of the ensemble learner to data streams from B. Wang and J. Pineau";
    }
    
    private static final long serialVersionUID = 1L;
    
    public IntOption randSeedOption = new IntOption("seed", 'r',
            "Seed for random behaviour of the classifier.", 1);

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "meta.AdaptiveRandomForest");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
        "The size of the ensemble.", 10, 1, Integer.MAX_VALUE);        
    
//    public IntOption samplingRateOption = new IntOption("samplingRate", 'i',
//            "The sampling rate of the positive instances.", 1, 1, 10);
    
	public IntOption kNNOption = new IntOption("kNN", 'k',
			"k-Nearest-Neighbor microcluster of the current instance to form a marcocluster", 5, 1,
			Integer.MAX_VALUE);
    
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
    
    public FlagOption disableDriftDetectionOption = new FlagOption("disableDriftDetection", 'd',
            "Should use ADWIN as drift detector?");

    protected Classifier baseLearner;
    protected int nEstimators;    
//    protected int samplingRate;
    protected boolean driftDetection;        
    protected ArrayList<Classifier> ensemble;   
    protected ArrayList<ADWIN> adwinEnsemble;  
    protected Instances[] pastSamples;
    protected SamoaToWekaInstanceConverter samoaToWeka = new SamoaToWekaInstanceConverter();
    
	protected double[] classSizeEstimation; // time-decayed size of each class
	protected double[] b;
    
    @Override
    public void resetLearningImpl() {
        // Reset attributes
    	this.baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
    	this.baseLearner.resetLearning();
        this.nEstimators = this.ensembleSizeOption.getValue(); 
//        this.samplingRate = this.samplingRateOption.getValue();
        this.driftDetection = !this.disableDriftDetectionOption.isSet();                
        this.ensemble = new ArrayList<Classifier>();
        if (this.driftDetection) {
        	this.adwinEnsemble = new ArrayList<ADWIN>();
        }
        for (int i = 0; i < this.nEstimators; i++) {
        	this.ensemble.add(this.baseLearner.copy());         	        
        	if (this.driftDetection) {
        		this.adwinEnsemble.add(new ADWIN());
        	}        	
		}
        this.pastSamples = null;
        this.randomSeed = this.randSeedOption.getValue();
        this.classifierRandom = new Random(this.randomSeed);
        
        this.classSizeEstimation = null;
        this.b = null;
    }

    @Override
    public void trainOnInstanceImpl(Instance instance) {        
        if(this.ensemble.isEmpty()) {
        	resetLearningImpl();
        }  
        adjustEnsembleSize(instance.numClasses());
        if (this.pastSamples == null) {
        	this.pastSamples = new Instances[instance.numClasses()];
        	for (int i = 0; i < this.pastSamples.length; ++i) {
        		this.pastSamples[i] = instance.dataset();
        		this.pastSamples[i].setClassIndex(this.pastSamples[i].numAttributes() - 1);
        	}
        }
        
        this.pastSamples[(int) instance.classValue()].add(instance);
        this.updateClassSize(instance);
        
        double lambda = 1.0;
        boolean changeDetected = false;        
        
        for (int i = 0 ; i < this.ensemble.size(); i++) {
        	double a = (double)(i + 1) / (double)this.nEstimators;
        	double samplingRate = this.calculatePoissonLambda(instance);
        	if (instance.classValue() == this.getMinorityClass()) {
        		lambda = a * samplingRate;
        		double lambdaSMOTE = (1 - a) * samplingRate;               	        								
				double k = MiscUtils.poisson(lambda, this.classifierRandom);
				if (k > 0) {
					for (int b = 0; b < k; b++) {
						this.ensemble.get(i).trainOnInstance(instance);					
					}
				}
				double kSMOTE = MiscUtils.poisson(lambdaSMOTE, this.classifierRandom);				
				if (kSMOTE > 0) {
					for (int b = 0; b < kSMOTE; b++) {
						Instance instanceSMOTE = onlineSMOTE((int) instance.classValue());						
						this.ensemble.get(i).trainOnInstance(instanceSMOTE);															
					}
				}
        	}
        	else {
        		double k = MiscUtils.poisson(lambda, this.classifierRandom);        		
        		if (k > 0) {
					for (int b = 0; b < k; b++) {
						this.ensemble.get(i).trainOnInstance(instance);					
					}
				}
        	}	
			if (this.driftDetection) {
				double pred = Utils.maxIndex(this.ensemble.get(i).getVotesForInstance(instance));
				double errorEstimation = this.adwinEnsemble.get(i).getEstimation();
				double inputValue = pred == instance.classValue() ? 1.0 : 0.0;
				boolean resInput = this.adwinEnsemble.get(i).setInput(inputValue);
				if (resInput) {
					if (this.adwinEnsemble.get(i).getEstimation() > errorEstimation) {
						changeDetected = true;
					}
				}
			}
		}
        
        if (changeDetected && this.driftDetection) {
        	double maxThreshold = 0.0;
        	int iMax = -1;
        	for (int i = 0; i < this.ensemble.size(); i++) {
				if (maxThreshold < this.adwinEnsemble.get(i).getEstimation()) {
					maxThreshold = this.adwinEnsemble.get(i).getEstimation();
					iMax = i;
				}
			}
        	if (iMax != -1) {
        		this.ensemble.get(iMax).resetLearning();
        		this.adwinEnsemble.set(iMax,new ADWIN());
        	}
        }     
    }

    @Override
    public double[] getVotesForInstance(Instance instance) {
        Instance testInstance = instance.copy();        
        DoubleVector combinedVote = new DoubleVector();

        for(int i = 0 ; i < this.ensemble.size() ; ++i) {
            DoubleVector vote = new DoubleVector(this.ensemble.get(i).getVotesForInstance(testInstance));
            if (vote.sumOfValues() > 0.0) {                                                                                                                               
            	vote.normalize();
                combinedVote.addValues(vote);                
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder arg0, int arg1) {
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }
    
    protected void adjustEnsembleSize(int nClasses) {
    	if (nClasses > this.nEstimators) {
    		for (int i = this.nEstimators; i < nClasses; i++) {
    			this.ensemble.add(this.baseLearner.copy()); 
    			this.nEstimators ++;
    			if (this.driftDetection) {
            		this.adwinEnsemble.add(new ADWIN());
            	}  			
			}
    	}
    }
    
    protected Instance onlineSMOTE(int classValue) {
    	int k = this.kNNOption.getValue();
    	if (this.pastSamples[classValue].numInstances() > 1) {
    		Instance x = this.pastSamples[classValue].instance(this.pastSamples[classValue].numInstances() - 1);    		
    		NearestNeighbourSearch search = new LinearNNSearch(this.pastSamples[classValue]);
    		try {
				Instances neighbours = search.kNearestNeighbours(x,Math.min(k,this.pastSamples[classValue].numInstances()-1));
				// create synthetic sample    	
				double[] values = new double[this.pastSamples[classValue].numAttributes()];
				int nn = this.classifierRandom.nextInt(neighbours.numInstances());
				Enumeration attrEnum = this.samoaToWeka.wekaInstance(this.pastSamples[classValue].instance(0)).enumerateAttributes();
				while(attrEnum.hasMoreElements()) {
					Attribute attr = (Attribute) attrEnum.nextElement();				
					if (!attr.equals(this.samoaToWeka.wekaInstance(this.pastSamples[classValue].instance(0)).classAttribute())) {
						if (attr.isNumeric()) {
							double dif = this.samoaToWeka.wekaInstance(neighbours.instance(nn)).value(attr) - this.samoaToWeka.wekaInstance(x).value(attr);
							double gap = this.classifierRandom.nextDouble();
							values[attr.index()] = (double) (this.samoaToWeka.wekaInstance(x).value(attr) + gap * dif);
						} else if (attr.isDate()) {
							double dif = this.samoaToWeka.wekaInstance(neighbours.instance(nn)).value(attr) - this.samoaToWeka.wekaInstance(x).value(attr);
							double gap = this.classifierRandom.nextDouble();
							values[attr.index()] = (long) (this.samoaToWeka.wekaInstance(x).value(attr) + gap * dif);
						} else {
							int[] valueCounts = new int[attr.numValues()];
							int iVal = (int) this.samoaToWeka.wekaInstance(x).value(attr);
							valueCounts[iVal]++;
							for (int nnEx = 0; nnEx < neighbours.numInstances(); nnEx++) {
								int val = (int) this.samoaToWeka.wekaInstance(neighbours.instance(nnEx)).value(attr);
								valueCounts[val]++;
							}
							int maxIndex = 0;
							int max = Integer.MIN_VALUE;
							for (int index = 0; index < attr.numValues(); index++) {
								if (valueCounts[index] > max) {
									max = valueCounts[index];
									maxIndex = index;
								}
							}
							values[attr.index()] = maxIndex;
						}
					} 
				}								
				values[this.pastSamples[classValue].classIndex()] = x.classValue();
								   		
	    		int[] indexValues = new int[x.numAttributes()];
	    		for (int i = 0; i < x.numAttributes(); i ++) {
	    			indexValues[i] = i;
	    		}
		      	
				//new synthetic instance
				Instance synthetic = x.copy();
				synthetic.addSparseValues(indexValues, values, x.numAttributes());
								
				return synthetic;
				
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();				
				return x;
			}    		
    	}
    	else {
    		return this.pastSamples[classValue].instance(this.pastSamples[classValue].numInstances() - 1);
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
}
