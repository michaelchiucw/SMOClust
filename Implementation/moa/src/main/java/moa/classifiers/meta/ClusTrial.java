package moa.classifiers.meta;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.cluster.Clustering;
import moa.clusterers.Clusterer;
import moa.clusterers.macro.dbscan.DBScan;
import moa.core.Measurement;
import moa.evaluation.BasicClassificationPerformanceEvaluator.Estimator;
import moa.options.ClassOption;

public class ClusTrial extends AbstractClassifier implements MultiClassClassifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public IntOption targetClassOption = new IntOption("targetClass", 't',
			"target Class", 1, 0, 1);
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "trees.HoeffdingTree -l NB");
	
	public ClassOption clusteringMethodOption = new ClassOption("clusteringMethod", 'c',
			"Clustering method to use.", Clusterer.class, "denstream.WithDBSCAN");
	
	public FloatOption epsilonOption = new FloatOption("epsilon", 'e',
			"Defines the epsilon neighbourhood", 0.16, 0, 1);
	
	public IntOption muOption = new IntOption("mu", 'm', "", 1, 0,
			Integer.MAX_VALUE);
	
	protected Classifier baseLearner;
	
	protected Clusterer clusterer;
	
	private int targetClass;
	
	// [0]: precision; [1]: recall
	private final int NUM_OF_METRICS = 2;
	private final double ALPHA = 0.999;
	private FadingFactorEstimator[] microStats;
	private FadingFactorEstimator[] marcoStats;
	
	
	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return this.baseLearner.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		this.baseLearner = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		if (this.baseLearner instanceof WEKAClassifier) {
			((WEKAClassifier) this.baseLearner).buildClassifier();
		}
		this.baseLearner.resetLearning();
		
		this.clusterer = ((Clusterer) getPreparedClassOption(this.clusteringMethodOption)).copy();
		this.clusterer.resetLearning();
		
		this.targetClass = this.targetClassOption.getValue();
		
		this.microStats = new FadingFactorEstimator[NUM_OF_METRICS];
		this.marcoStats = new FadingFactorEstimator[NUM_OF_METRICS];
		for (int i = 0; i < NUM_OF_METRICS; ++i) {
			this.microStats[i] = new FadingFactorEstimator(ALPHA);
			this.marcoStats[i] = new FadingFactorEstimator(ALPHA);
		}
		
	}
	
	private void updateMicroStats(double predictedClass, double trueClass) {
		if (predictedClass == this.targetClass) { // Precision
			this.microStats[0].add(predictedClass == trueClass ? 1.0 : 0.0);
		}
		
		if (trueClass == this.targetClass) { // Recall
			this.microStats[1].add(predictedClass == trueClass ? 1.0 : 0.0);
		}
	}
	
	private void updateMarcoStats(double predictedClass, double trueClass) {
		if (predictedClass == this.targetClass) { // Precision
			this.marcoStats[0].add(predictedClass == trueClass ? 1.0 : 0.0);
		}
		
		if (trueClass == this.targetClass) { // Recall
			this.marcoStats[1].add(predictedClass == trueClass ? 1.0 : 0.0);
		}
	}
	
	private double calF1(double precision, double recall, double beta) {
		if (precision == 0.0 && recall == 0.0) {
			return 0.0;
		}
		double beta_square = beta * beta;
		return (1 + beta_square) * ((precision * recall) / (beta_square * precision + recall));
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {

		double trueClass = inst.classValue();
		
		this.baseLearner.trainOnInstance(inst);
		
		// Filter out the class attribute
		Instance noClassInst = new DenseInstance(inst);
		noClassInst.deleteAttributeAt(inst.classIndex());
		
		if (this.clusterer.trainingHasStarted()) {
			Clustering microClusters = this.clusterer.getMicroClusteringResult();
			
			System.out.println("True Label: " + inst.classValue());
			System.out.println("time step: " + this.trainingWeightSeenByModel);
			if (microClusters != null) {
				System.out.println("# Micro Clusters: " + microClusters.size());
				double microIn = microClusters.getMaxInclusionProbability(noClassInst);
				System.out.println("In microClusters of class " + this.targetClass + "? " + microIn);
				if (this.targetClass == 0.0) {
					this.updateMicroStats(microIn == 1.0 ? 0.0 : 1.0, trueClass);
				} else if (this.targetClass == 1.0) {
					this.updateMicroStats(microIn == 1.0 ? 1.0 : 0.0, trueClass);
				} else {
					System.err.println("Micro: Not Class 0.0 not Class 1.0");
				}
				System.out.println("Micro Precision: " + this.microStats[0].estimation());
				System.out.println("Micro Precision B: " + this.microStats[0].getB());
				System.out.println("Micro Recall: " + this.microStats[1].estimation());
				System.out.println("Micro Recall B: " + this.microStats[1].getB());
				System.out.println("Micro balanced F1: " + this.calF1(this.microStats[0].estimation(), this.microStats[1].estimation(), 1));
				System.out.println("===============================================");
			}
			
			DBScan dbscan = new DBScan(microClusters, this.epsilonOption.getValue(), this.muOption.getValue());
			Clustering marcoClusters = dbscan.getClustering(microClusters);
//			Clustering marcoClusters = KMeans.gaussianMeans(dbClustering, microClusters);
			
			if (marcoClusters != null) {
				System.out.println("# Marco Clusters: " + marcoClusters.size());
				double marcoIn = marcoClusters.getMaxInclusionProbability(noClassInst);
				System.out.println("In marcoClusters of class " + this.targetClass + "? " + marcoIn);
				if (this.targetClass == 0.0) {
					this.updateMarcoStats(marcoIn == 1.0 ? 0.0 : 1.0, trueClass);
				} else if (this.targetClass == 1.0) {
					this.updateMarcoStats(marcoIn == 1.0 ? 1.0 : 0.0, trueClass);
				} else {
					System.err.println("Marco: Not Class 0.0 not Class 1.0");
				}
				System.out.println("Marco Precision: " + this.marcoStats[0].estimation());
				System.out.println("Marco Precision B: " + this.marcoStats[0].getB());
				System.out.println("Marco Recall: " + this.marcoStats[1].estimation());
				System.out.println("Marco Recall B: " + this.marcoStats[1].getB());
				System.out.println("Marco balanced F1: " + this.calF1(this.marcoStats[0].estimation(), this.marcoStats[1].estimation(), 1));
				System.out.println("===============================================");
			}
			
//			Clustering clusRes = this.clusterer.getClusteringResult();
//			
//			System.out.println("True Label: " + inst.classValue());
//			System.out.println("# Marco Clusters: " + clusRes.size());
//			System.out.println("In marcoClusters of class 1.0? " + clusRes.getMaxInclusionProbability(noClassInst));
			
		}
		
		if (inst.classValue() == this.targetClass) {
			this.clusterer.trainOnInstance(noClassInst);
		}
		
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return new Measurement[] {
				new Measurement("Micro Clustering Class " + this.targetClass + " Precision", this.microStats[0].estimation()),
				new Measurement("Micro Clustering Class " + this.targetClass + " Recall", this.microStats[1].estimation()),
				new Measurement("Micro Clustering Class " + this.targetClass + " Balanced F1", this.calF1(this.microStats[0].estimation(), this.microStats[1].estimation(), 1)),
				new Measurement("Marco Clustering Class " + this.targetClass + " Precision", this.marcoStats[0].estimation()),
				new Measurement("Marco Clustering Class " + this.targetClass + " Recall", this.marcoStats[1].estimation()),
				new Measurement("Marco Clustering Class " + this.targetClass + " Balanced F1", this.calF1(this.marcoStats[0].estimation(), this.marcoStats[1].estimation(), 1))
		};
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub

	}
	
    public class FadingFactorEstimator implements Estimator {

        protected double alpha;

        protected double estimation;

        protected double b;

        public FadingFactorEstimator(double a) {
            alpha = a;
            estimation = 0.0;
            b = 0.0;
        }

        @Override
        public void add(double value) {
        	if(!Double.isNaN(value)) {
        		estimation = alpha * estimation + value;
                b = alpha * b + 1.0;
        	}
        }

        @Override
        public double estimation() {
            return b > 0.0 ? estimation / b : 0;
        }
        
        public double getB() {
        	return b;
        }

    }

}
