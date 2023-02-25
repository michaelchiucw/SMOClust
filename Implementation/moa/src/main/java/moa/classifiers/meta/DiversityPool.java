/*
 *    DiversityPool.java
 *    Copyright (C) 2018 University of Leicester, Leicester, United Kingdom
 *    @author Chun Wai Chiu (cwc13@leicester.ac.uk)
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
package moa.classifiers.meta;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.core.diversitytest.DiversityParallelTester;
import moa.classifiers.core.diversitytest.DiversityTest;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;

public class DiversityPool extends AbstractClassifier implements MultiClassClassifier {

	/**
	 * Default serial version
	 */
	private static final long serialVersionUID = 1L;
	
	public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "The Base Learner.", Classifier.class, "trees.HoeffdingTree -l NB");
	
	public IntOption poolSizeOption = new IntOption("maxPooSize:", 's',
			"The maximum size of the classifier pool.", 10, 1, Integer.MAX_VALUE);
		
	public IntOption testChunkSizeOption = new IntOption("testChunkBufferSize", 'b',
			"The size of the sliding window (buffer) for testing after concept drift.", 400, 0, Integer.MAX_VALUE);
	
	public ClassOption driftDetectorOption = new ClassOption("driftDetector", 'd',
            "Drift detection method to use.", ChangeDetector.class, "DDM");
	
	public ClassOption diversityTestOption = new ClassOption("diversityTest", 't',
            "Diversity test to be used.", DiversityTest.class, "QStatistics");
	
	/**
	 * The array to store classifiers.
	 */
	private List<Classifier> classifierPool;
	/**
	 * Weights for weighted majority vote.
	 */
	protected double[] weights;
	protected double newClassifierWeight;
	
	protected Classifier currentClassifier;
    protected Classifier newClassifier;

    protected ChangeDetector driftDetector;
    
    protected DRIFT_LEVEL drift_level;
    
    protected boolean warningZoneReset;
    
    protected boolean ensembleMode;
    
    protected double warningDetected;
    protected double changeDetected;
    
    protected List<Instance> testChunk;
    
	@Override
	public String getPurposeString() {
		return "A classifier that can learn with recurring concept drift data stream.";
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		out.append("Number of classifiers in pool (including the recent classifier): " + classifierPool.size() + "\n");
		out.append(((this.ensembleMode) ? "Ensemble Mode!" : "") + "\n");
		out.append("Current Model is: " + this.classifierPool.indexOf(this.currentClassifier) + "\n");
	}

	@Override
	public Classifier[] getSubClassifiers() {
		return this.classifierPool.toArray(new Classifier[this.classifierPool.size()]);
	}
	
	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		List<Measurement> measurementList = new LinkedList<Measurement>();
		measurementList.add(new Measurement("Warning Detected", this.warningDetected));
		measurementList.add(new Measurement("Change Detected", this.changeDetected));
		
		return measurementList.toArray(new Measurement[measurementList.size()]);
	}

	@Override
	public void resetLearningImpl() {
		this.currentClassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
		if (this.currentClassifier instanceof WEKAClassifier) {
			((WEKAClassifier) this.currentClassifier).buildClassifier();
		}
		
		
        this.newClassifier = this.currentClassifier.copy();
        
        this.currentClassifier.resetLearning();
        this.newClassifier.resetLearning();
        
        this.driftDetector = ((ChangeDetector) getPreparedClassOption(this.driftDetectorOption)).copy();
        
        this.classifierPool = new ArrayList<Classifier>();
        this.classifierPool.add(this.currentClassifier);
        
        this.weights = new double[this.poolSizeOption.getValue()];
        this.newClassifierWeight = 0.0;
        
        this.testChunk = new ArrayList<Instance>(this.testChunkSizeOption.getValue());
        
        this.changeDetected = 0;
        this.warningDetected = 0;
        this.warningZoneReset = false;
        this.ensembleMode = false;
        this.drift_level = DRIFT_LEVEL.NORMAL;
	}
	
	@Override
	public double[] getVotesForInstance(Instance instance) {
		
		if (this.ensembleMode && this.trainingHasStarted()) {
			/**
			 * Weighted Majority Vote
			 * "this.classifierPool.size()+1" because the new classifier has joined the W.M.V.
			 */
			DoubleVector combinedVote = new DoubleVector();
			
			for (int i = 0; i < this.classifierPool.size(); ++i) {
				if (this.weights[i] > 0.0) {
					DoubleVector vote = new DoubleVector(this.classifierPool.get(i).getVotesForInstance(instance));
					if (vote.sumOfValues() > 0.0) {
						vote.normalize();
						/**
						 * scale weight and prevent overflow.
						 */
						vote.scaleValues(this.weights[i] / (1.0 * (this.classifierPool.size()+1) + 1.0));
						combinedVote.addValues(vote);
					}
				}
			}
			if (this.newClassifierWeight > 0.0) {
				DoubleVector vote = new DoubleVector(this.newClassifier.getVotesForInstance(instance));
				if (vote.sumOfValues() > 0.0) {
					vote.normalize();
					/**
					 * scale weight and prevent overflow.
					 */
					vote.scaleValues(this.newClassifierWeight / (1.0 * (this.classifierPool.size()+1) + 1.0));
					combinedVote.addValues(vote);
				}
			}
			return combinedVote.getArrayRef();
		} else {
			/**
			 * Current Classifier Vote
			 */
			return this.currentClassifier.getVotesForInstance(instance);
		}
		
	}

	@Override
	public void trainOnInstanceImpl(Instance instance) {
		
		if (this.ensembleMode) {
			this.ensembleModeAutoSwitch(instance);
		} else {
			/**
			 * DDM tracks the error rate on the current classifier but NOT in ensemble mode.
			 */
			double prediction = this.currentClassifier.correctlyClassifies(instance) ? 0.0 : 1.0;
//			double prediction = this.correctlyClassifies(instance) ? 0.0 : 1.0;
			this.driftDetector.input(prediction);
			this.drift_level = DRIFT_LEVEL.NORMAL;
			if (this.driftDetector.getChange()) {
				this.drift_level = DRIFT_LEVEL.OUTCONTROL;
			} else if (this.driftDetector.getWarningZone()) {
				this.drift_level = DRIFT_LEVEL.WARNING;
			}
			
			switch (this.drift_level) {
				case OUTCONTROL:
//					System.out.println("OUTCONTROL");
					if (this.classifierPool.size() < this.poolSizeOption.getValue()) {
						/**
						 *  If the pool is not full, then simply add the new classifier.
						 */
//						System.out.println("A");
						this.currentClassifier = this.newClassifier;
						classifierPool.add(this.currentClassifier);
						
						this.ensembleMode = false;
					} else {
						/**
						 *  If the pool is full.
						 */
						/**
						 * Weight the pool and the new classifier with the test chunk, prepare for the W.M.V.
						 */
						if (this.testChunk.size() <= this.testChunkSizeOption.getValue()) {
							this.clearWeights();
							for (Instance example : this.testChunk) {
								this.weightClassifiers(example);
							}
						}
						this.ensembleModeAutoSwitch(instance);
						
					}
					/**
					 *  Create a new classifier ready for next alarm.
					 */
					this.newClassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
					if (this.newClassifier instanceof WEKAClassifier) {
						((WEKAClassifier) this.newClassifier).buildClassifier();
					}
					this.newClassifier.resetLearning();
					
					
//					System.out.println((this.ensembleMode) ? "Ensemble Mode!" : "index of current classifier: " + this.classifierPool.indexOf(this.currentClassifier) + " | pool size: " + this.classifierPool.size());
				
					this.changeDetected++;

					break;
				case WARNING:
//					System.out.println("WARNING");
		    			if (this.warningZoneReset) {
		    				this.newClassifier.resetLearning();
		    				this.warningZoneReset = false;
		    				
		    				this.testChunk.clear();
		    				this.clearWeights();
		    			}
		    			/**
		    			 *  Start accumulating examples when reaching warning level.
		    			 */
		    			if (this.testChunk.size() >= this.testChunkSizeOption.getValue()) {
		    				this.testChunk.remove(0);
		    			}
		    			this.testChunk.add(instance);
		    			
		    			/**
		    			 *  Start training the new classifier for the new concept.
		    			 */
		    			this.newClassifier.trainOnInstance(instance);
		    		
		    			this.warningDetected++;
					break;
				case NORMAL:
					// System.out.println("NORMAL");
					
		    			this.warningZoneReset = true;
					break;
				default:
					System.out.println("ERROR!");
					break;
			}
			
		    this.currentClassifier.trainOnInstance(instance);
			
		}

	}
	
	private void ensembleModeAutoSwitch(Instance instance) {
		if (this.testChunk.size() < this.testChunkSizeOption.getValue()) {
			/**
			 * If the test chunk is still not full, keep weighting and normalizing the classifiers for the Weighted Majority Votes.
			 * The new classifier has to be trained in parallel.
			 */
//			System.out.println("testChunk is NOT full | Weighting | Ensemble Mode: ON");
			this.weightClassifiers(instance);
			this.testChunk.add(instance);
			this.newClassifier.trainOnInstance(instance);
			this.ensembleMode = true;
		} else {
			/**
			 * When the test chunk is full, trigger the model selection mechanism in the pool.
			 * If none of the old classifiers passes the threshold, then trigger the diversity mechanism to discard an old model and add the new one.
			 */
//			System.out.println("testChunk is full | Model selection | Ensemble Mode: OFF");
			Classifier closestModel = this.getClosestClassifierAfterDrift(this.testChunk, this.newClassifier);
			if (closestModel != null) {
//				System.out.println("index of closest: " + this.classifierPool.indexOf(closestModel));
				this.currentClassifier = closestModel;
			} else {
//				System.out.println("Diversity | currentClassifier is the new classifier");
				this.discardModelByDiversity();
				this.currentClassifier = this.newClassifier;
				classifierPool.add(this.currentClassifier);
			}
			/**
			 *  Create a new classifier ready for next alarm.
			 */
			this.newClassifier = ((Classifier) getPreparedClassOption(this.baseLearnerOption)).copy();
			if (this.newClassifier instanceof WEKAClassifier) {
				((WEKAClassifier) this.newClassifier).buildClassifier();
			}
			this.newClassifier.resetLearning();
			
			this.ensembleMode = false;
		}
	}
	
	private Classifier getClosestClassifierAfterDrift(List<Instance> chunk, Classifier newModel) {
		double[] evalWeights = new double[this.classifierPool.size()];
		double newModelAccuracy = 0.0;
		
		int best = 0;
		for (int i = 0; i < this.classifierPool.size(); ++i) {
			if (i != this.classifierPool.indexOf(this.currentClassifier)) {
				for (Instance example : chunk) {
					evalWeights[i] += this.classifierPool.get(i).correctlyClassifies(example) ? 1.0 : 0.0;
				}
				evalWeights[i] /= chunk.size();
				if (evalWeights[i] > evalWeights[best]) {
					best = i;
				}
			}
		}
		
		/**
		 * Calculate the accuracy of the new classifier.
		 */
		for (Instance example : chunk) {
			newModelAccuracy += newModel.correctlyClassifies(example) ? 1.0 : 0.0;
		}
		newModelAccuracy /= chunk.size();
		
		/**
		 * Calculate the S.D. of the best model from the pool.
		 */
		double sd = 0.0;
		for (Instance example : chunk) {
			double pred = this.classifierPool.get(best).correctlyClassifies(example) ? 1.0 : 0.0;
			sd += (pred - evalWeights[best]) * (pred - evalWeights[best]);
		}
		sd /= chunk.size();
		sd = Math.sqrt(sd);
		double standardError = sd / Math.sqrt(chunk.size());

//		System.out.println("Best: " + evalWeights[best]);
//		System.out.println("newModel: " + newModelAccuracy);
//		System.out.println("range: " + evalWeights[best] + " +/- " + 1.96*standardError);

		if (newModelAccuracy >= (evalWeights[best] - 1.96*standardError) && newModelAccuracy <= (evalWeights[best] + 1.96*standardError)) {
//			System.out.println("Change");
			return this.classifierPool.get(best);
		}
		return null;
	}
	
	private void discardModelByDiversity() {
		
		DiversityTest test = (DiversityTest) getPreparedClassOption(this.diversityTestOption);
		DiversityParallelTester tester = new DiversityParallelTester(this.testChunk, this.classifierPool, test);
		Classifier toDelete = tester.getSuggestedDeletion();
		
//		int index = this.classifierPool.indexOf(toDelete);
//		System.out.println("Deletion suggest: " + index);
		
		this.classifierPool.remove(toDelete);
	}
	
	private void weightClassifiers(Instance inst) {
		for (int i = 0; i < this.classifierPool.size(); ++i) {
			this.weights[i] += this.classifierPool.get(i).correctlyClassifies(inst) ? 1.0 : 0.0;
		}
		this.newClassifierWeight += this.newClassifier.correctlyClassifies(inst) ? 1.0 : 0.0;
	}
	
	private double[] getNormalisedWeigths() {
		double[] normalisedWeights = null;
		if (this.testChunk.size() > 0) {
			normalisedWeights = this.weights.clone();
			for (int i = 0; i < normalisedWeights.length; ++i) {
				normalisedWeights[i] /= this.testChunk.size();
			}
		} else {
			System.out.println("The test chunk is empty, cannot normalise the weigths.");
		}
		return normalisedWeights;
	}
	
	private void clearWeights() {
		this.weights = new double[this.poolSizeOption.getValue()];
	}
	
	enum DRIFT_LEVEL {
		NORMAL, WARNING, OUTCONTROL
	}
}