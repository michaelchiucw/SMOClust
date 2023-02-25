package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;
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
import moa.cluster.CFCluster;
import moa.cluster.Clustering;
import moa.cluster.SphereCluster;
import moa.clusterers.Clusterer;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.core.Utils;
import moa.options.ClassOption;

public class SMOClust extends AbstractClassifier implements MultiClassClassifier {
	
	private static final long serialVersionUID = 1L;
	
	public IntOption randSeedOption = new IntOption("seed", 'i',
            "Seed for random behaviour of the classifier.", 1);
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "meta.OzaBag");
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	public ClassOption clusteringMethodOption = new ClassOption("clusteringMethod", 'x',
			"Clustering method to use.", Clusterer.class, "clustream.Clustream");
	
	public FloatOption gaussianNoiseVarianceOption = new FloatOption("gaussianNoiseVariance", 'v',
			"Variance for Gaussian noise to create synthetic examples", 0.01, 0.0, Double.MAX_VALUE);
	
	public FloatOption categoricalChangeProbabilityOption = new FloatOption("categoricalChangeProbability", 'c',
			"Probability for categorical attributes to change to another value.", 0.2, 0.0, 1.0);
	
	public IntOption kNNOption = new IntOption("kNN", 'k',
			"k-Nearest-Neighbor microcluster of the current instance to form a marcocluster", 5, 1,
			Integer.MAX_VALUE);
	
	public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");
	
	public FlagOption disableDDMOption = new FlagOption("disableDDM", 'z', "disableDDM");
	
	protected BaseLearner baseLearner;
	
	protected SamoaToWekaInstanceConverter moaToWekaInstanceConverter;
	protected WekaToSamoaInstanceConverter wekaToMoaInstanceConverter;
	
	protected Clusterer[] clusterers;
	protected Instance[] last_inst;
	
	protected ChangeDetector driftDetector;
	protected DRIFT_LEVEL drift_level;
	protected double[] timeStepsAfterDrift;
	protected int driftCount;

    @Override
    public String getPurposeString() {
        return "SMOClust";
    }
	
	@Override
    public void resetLearningImpl() {
		this.randomSeed = this.randSeedOption.getValue();
		this.classifierRandom = new Random(this.randomSeed);
		
		this.baseLearner = new BaseLearner(((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy());
		
		this.moaToWekaInstanceConverter = new SamoaToWekaInstanceConverter();
		this.wekaToMoaInstanceConverter = new WekaToSamoaInstanceConverter();
		
		this.clusterers = null;
		this.last_inst = null;
		
		this.driftDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectorOption)).copy();
		this.driftCount = 0;
		this.timeStepsAfterDrift = null;
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
		if (this.clusterers == null) {
			this.clusterers = new Clusterer[inst.numClasses()];
			for (int i = 0; i < this.clusterers.length; ++i) {
				clusterers[i] = ((Clusterer) getPreparedClassOption(this.clusteringMethodOption)).copy();
				clusterers[i].resetLearning();
			}
		}
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
		
		/**
		 * Train ensemble
		 */
		this.baseLearner.trainOnInstance(inst);
		this.last_inst[current_class] = inst.copy();
		
		int maj_class = this.baseLearner.getMajorityClass();
		int min_class = this.baseLearner.getMinorityClass();
		
		Clustering[] mClusteringResults = null;
		double[][][] allkNNmCluster_index_distance = null;
		
		mClusteringResults = new Clustering[inst.numClasses()];
		allkNNmCluster_index_distance = new double[inst.numClasses()][][]; // [class][sorted index by distance][0 || 1]; [][0] is clusterIndex; [][1] is the distance;
		
		for (int i = 0; i < mClusteringResults.length; ++i) {
			mClusteringResults[i] = this.clusterers[i].getMicroClusteringResult();
		}
		
		Instance last_minority_oneHot = null;
		if (this.last_inst[min_class] != null) {
			last_minority_oneHot = this.last_inst[min_class].copy();
			try {
				last_minority_oneHot = this.nominalToBinary(last_minority_oneHot);
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		
		boolean isMicroClustersReady = this.checkMicroClustersReady(mClusteringResults);
		
		while (this.baseLearner.getRawClassSize(min_class) < this.baseLearner.getRawClassSize(maj_class) && (isMicroClustersReady || this.last_inst[min_class] != null)) {
			if (isMicroClustersReady) {
				
				CFCluster anchor_mCluster = null;
				anchor_mCluster = this.getAnchor_mCluster(mClusteringResults[min_class]);
				for (int i = 0; i < allkNNmCluster_index_distance.length; ++i) {
					allkNNmCluster_index_distance[i] = this.kNNmClusterIndex(anchor_mCluster, mClusteringResults[i]);
				}
				boolean isSurroundedBySameClass = checkSurroundedBySameClass(allkNNmCluster_index_distance, min_class);
				
				try {
					Instance synthInstOneHot = isSurroundedBySameClass ? this.generateSynthInstFromkNN(mClusteringResults, allkNNmCluster_index_distance, anchor_mCluster, min_class, last_minority_oneHot.dataset()) : 
																		 this.generateSynthInstByGaussianSamplingmCluster(anchor_mCluster, min_class, last_minority_oneHot.dataset());
					Instance synthInst = this.binaryToNominal(synthInstOneHot, this.last_inst[min_class].dataset());
					
					synthInstOneHot.deleteAttributeAt(synthInstOneHot.classIndex());
    				this.clusterers[min_class].trainOnInstance(synthInstOneHot, false);
    				
    				// Train base learner with synthetic example.
            		this.baseLearner.trainOnInstance(synthInst);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(1);
					break;
				}
			} else {
				/**
    			 * if (mClusteringResult/kNNmCluster_indices == null || kNNmCluster_indices < (this.muOption.getValue() - 1)
    			 * Create a synthetic instance by adding gaussian noise to the most recent real instance.
    			 */
				Instance synthInst = this.addGaussianNoiseToInstance(this.last_inst[min_class]);
        		
        		// Train the corresponding stream clustering method with synthetic data
				try {
					Instance synthInstOneHot = this.nominalToBinary(synthInst);
					synthInstOneHot.deleteAttributeAt(synthInstOneHot.classIndex());
    				this.clusterers[min_class].trainOnInstance(synthInstOneHot, false);
				} catch (Exception e) {
					e.printStackTrace();
					System.exit(2);
					break;
				}
				// Train base learner with synthetic example.
        		this.baseLearner.trainOnInstance(synthInst);
			}
			
		} // End-while
		
		// Setting up the data for training the clusterer
		Instance to_train_clusterer = inst.copy();
		try {
			to_train_clusterer = this.nominalToBinary(to_train_clusterer);
		} catch (Exception e) {
			e.printStackTrace();
		}
		to_train_clusterer.deleteAttributeAt(to_train_clusterer.classIndex());
		// Train the corresponding stream clustering method with the most current real instance.
		this.clusterers[current_class].trainOnInstance(to_train_clusterer, true);
		
    }
	
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
	
	private Instance generateSynthInstFromkNN(Clustering[] mClusteringResults, double[][][] allkNNmCluster_index_distance, CFCluster anchor, int currentClass, Instances dataset) {
		Clustering mClusters = (Clustering) mClusteringResults[currentClass].copy();
		double[][] sorted_index_distance = Arrays.copyOf(allkNNmCluster_index_distance[currentClass], allkNNmCluster_index_distance[currentClass].length);
		double[] anchor_centre = anchor.getCenter();
		
		// Create a sphere cluster wrapper for the current inst. Let it be SphereCluster(inst).
		SphereCluster anchor_mCluster = new SphereCluster(anchor_centre, anchor.getRadius());

		SphereCluster[] mClusters_to_combine = new SphereCluster[sorted_index_distance.length+1];
		mClusters_to_combine[0] = anchor_mCluster;
		for (int i = 0; i < sorted_index_distance.length; ++i) {
			SphereCluster temp = (SphereCluster) mClusters.get((int) sorted_index_distance[i][0]).copy();
			mClusters_to_combine[i+1] = temp;
		}
		
		SphereCluster resultingCluster = this.combine_multiple_mClusters(mClusters_to_combine);
		
		// Sample the resulting SphereCluster as the synthetic example.
		Instance sample_inst = resultingCluster.sample_around_target(classifierRandom, anchor_centre);
		double[] sample = sample_inst.toDoubleArray();
		
		// Append the Class label to the synthetic example.
		double[] sampleWithClass = new double[sample.length+1];
		System.arraycopy(sample, 0, sampleWithClass, 0, sample.length);
		sampleWithClass[sampleWithClass.length-1] = (double) currentClass;
		
		// Create an Instance based on the array and set weight to 1.0;
		Instance synthInst = new DenseInstance(1d, sampleWithClass);
		synthInst.setDataset(dataset);
		
		return synthInst;
		
	}
	
	private SphereCluster combine_multiple_mClusters(SphereCluster[] mClusters) {
		double[][] all_centres = new double[mClusters.length][];
		double[] all_weights = new double[mClusters.length];
		double[] all_radius = new double[mClusters.length];
		
		double sum_of_weights = 0.0;
		
		for (int i = 0; i < all_centres.length; ++i) {
			all_centres[i] = mClusters[i].getCenter();
			all_weights[i] = mClusters[i].getWeight();
			sum_of_weights += all_weights[i];
			all_radius[i] = mClusters[i].getRadius();
		}
		
		int dim = all_centres[0].length;
		double[] newCentre = new double[dim];
		
		for (int i = 0; i < newCentre.length; ++i) { // i-th dimension
			double dim_result = 0.0;
			for (int j = 0; j < all_centres.length; ++j) { // j-th centre
				dim_result += all_centres[j][i] * all_weights[j];
			}
			newCentre[i] = dim_result / sum_of_weights;
		}
		
		double[] r_n = new double[mClusters.length];
		for (int i = 0; i < r_n.length; ++i) {
			r_n[i] = all_radius[i] + Math.abs(distance(all_centres[i], newCentre));
		}
		
		Arrays.sort(r_n);
		double newRadius = r_n[r_n.length-1];
		
		return new SphereCluster(newCentre, newRadius);
	}
	
	private double distance(double[] v1, double[] v2){
		double distance = 0.0;
		for (int i = 0; i < v1.length; i++) {
			double d = v1[i] - v2[i];
			distance += d * d;
		}
		return Math.sqrt(distance);
	}
	
	private Instance generateSynthInstByGaussianSamplingmCluster(CFCluster anchor, int current_class, Instances dataset) {

		Instance sample_inst = anchor.sampleGaussian(this.classifierRandom);
		double[] sample = sample_inst.toDoubleArray();
		
		// Append the Class label to the synthetic example.
		double[] sampleWithClass = new double[sample.length+1];
		System.arraycopy(sample, 0, sampleWithClass, 0, sample.length);
		sampleWithClass[sampleWithClass.length-1] = (double) current_class;
		
		// Create an Instance based on the array and set weight to 1.0;
		Instance synthInst = new DenseInstance(1d, sampleWithClass);
		synthInst.setDataset(dataset);
		
		return synthInst;
	}
	
	private CFCluster getAnchor_mCluster(Clustering mClusters) {
		double[] weights = new double[mClusters.size()];
		for (int i = 0; i < weights.length; ++i) {
			CFCluster tmp = (CFCluster) mClusters.get(i);
			weights[i] = tmp.getMeanTimeStamp() + 1.0; // just in case the Mean Time stamp is 0, then it will get 0 chance to be selected.
		}
		
		int index_to_get = MiscUtils.chooseRandomIndexBasedOnWeights(weights, this.classifierRandom);
		return (CFCluster) mClusters.get(index_to_get);
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
	
	private boolean checkMicroClustersReady(Clustering[] mClusteringResults) {
		for (Clustering mClusteringResult : mClusteringResults) {
			// k+1 because we need to skip the first element when find kNN for anchor mCluster.
			if (mClusteringResult == null || mClusteringResult.size() < this.kNNOption.getValue() + 1) {
				return false;
			}
		}
		return true;
	}
	
	private boolean checkSurroundedBySameClass(double[][][] allkNNmCluster_index_distance, int currentClass) {
		double minimum = Double.MAX_VALUE;
		int minIndex = -1;
		for (int i = 0; i < allkNNmCluster_index_distance.length; ++i) {
			double current_avgDistance = kNNAvgDistance(allkNNmCluster_index_distance[i]);
			if ((i == 0) || (current_avgDistance < minimum)) {
				minIndex = i;
				minimum = current_avgDistance;
			}
		}
		return minIndex == currentClass;
	}
	
	private double kNNAvgDistance(double[][] kNNmCluster_index_distance) {
		return kNNTotalDistance(kNNmCluster_index_distance) / kNNmCluster_index_distance.length;
	}
	
	private double kNNTotalDistance(double[][] kNNmCluster_index_distance) {
		double to_return = 0.0;
		for (double[] index_distance : kNNmCluster_index_distance) {
			to_return += index_distance[1];
		}
		return to_return;
	}

    private double[][] kNNmClusterIndex(CFCluster anchor, Clustering mClusteringResult) {
    	double[][] allDistanceWithClusterIndex = new double[mClusteringResult.size()][]; // [][0] is clusterIndex; [][1] is the distance;
    	
    	boolean result_contains_anchor = false;
    	for (int i = 0; i < mClusteringResult.size(); ++i) {
    		CFCluster mCluster = (CFCluster) mClusteringResult.get(i);
    		result_contains_anchor = anchor.equals(mCluster);
    		double distance = mCluster.getHullDistance(anchor);
    		allDistanceWithClusterIndex[i] = new double[] {i, distance <= 0 ? 0.0 : distance};
    		
    	}
    	Arrays.sort(allDistanceWithClusterIndex, Comparator.comparingDouble(ele -> ele[1])); // Sorting by distance in decending order.
    	int k = this.kNNOption.getValue();
    	double[][] to_return = new double[k][];
    	// Skip anchor itself if result contains it.
    	System.arraycopy(allDistanceWithClusterIndex, result_contains_anchor ? 1 : 0, to_return, 0, k);
    	return to_return;
    }
	
	private Instance binaryToNominal(Instance bin_inst, Instances original_header) throws Exception {
		Instance tmp_inst = new DenseInstance(original_header.numAttributes());
		tmp_inst.setDataset(original_header);
		
		for (int i = 0; i < original_header.numAttributes(); ++i) {
			String current_attr_name = original_header.attribute(i).name();
			ArrayList<Double> values = new ArrayList<Double>();
			
			for (int j = 0; j < bin_inst.numAttributes(); ++j) {
				if (bin_inst.attribute(j).name().contains(current_attr_name)) {
					values.add(bin_inst.value(j));
				}
			}
			
			if (values.size() == 1 && original_header.attribute(i).isNumeric()) { // Numeric attribute
				tmp_inst.setValue(i, values.get(0));
			} else if (values.size() == 1 && original_header.attribute(i).isNominal()) { // True Binary attribute
				int tmp_value = (int) Math.round(values.get(0));
				if (tmp_value >= 1) tmp_value = 1;
				if (tmp_value <= 0) tmp_value = 0;
				tmp_inst.setValue(i, tmp_value);
			} else if (values.size() > 1) { // Binary attribute: to be converted to nominal attribute
				
				double maxValue = Collections.max(values);
				int maxIndex = values.indexOf(maxValue);
				tmp_inst.setValue(i, maxIndex);
				
			} else { // matched_attributes.size == 0, which shouldn't happen but just in case.
				throw new Exception("No matched attribute.");
			}
		}
		
		return tmp_inst;
	}
	
	private Instance nominalToBinary(Instance original_inst) throws Exception {
		Instances moaInstances = new Instances(original_inst.dataset());
		moaInstances.add(original_inst);
		
		weka.core.Instances wekaInstances = this.moaToWekaInstanceConverter.wekaInstances(moaInstances);
		weka.filters.unsupervised.attribute.NominalToBinary nom2BinFilter = new weka.filters.unsupervised.attribute.NominalToBinary();
		nom2BinFilter.setInputFormat(wekaInstances);
		wekaInstances = weka.filters.Filter.useFilter(wekaInstances, nom2BinFilter);
		moaInstances = this.wekaToMoaInstanceConverter.samoaInstances(wekaInstances);
		
		return moaInstances.get(0);
	}
	
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