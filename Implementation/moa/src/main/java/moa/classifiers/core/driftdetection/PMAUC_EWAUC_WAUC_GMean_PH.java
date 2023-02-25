/*
 *    PMAUC_EWAUC_WAUC_GMean_PH.java
 *    Copyright (C) 2021 University of Birmingham, Birmingham, United Kingdom
 *    @author Chun Wai Chiu (cxc1015@student.bham.ac.uk / michaelchiucw@gmail.com)
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
 */
package moa.classifiers.core.driftdetection;

import java.util.Arrays;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;

import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;
import weka.core.Utils;

/**
 * 	PMAUC-PH, EWAUC-PH, WAUC-PH and GMean-PH from Wang et al. IJCNN 2020.
 *
 *  <p>WANG, S.; MINKU, L.; "AUC Estimation and Concept Drift Detection for Imbalanced Data Streams with Multiple Classes",
 *  IEEE International Joint Conference on Neural Networks (IJCNN), 8 pages, July 2020</p>
 *
 *  @author Chun Wai Chiu (cxc1015@student.bham.ac.uk / michaelchiucw@gmail.com)
 *  @version $Revision: 1 $
 */
public class PMAUC_EWAUC_WAUC_GMean_PH extends AbstractChangeDetector {
	
   public IntOption minNumInstancesOption = new IntOption(
            "minNumInstances",
            'n',
            "The minimum number of instances before permitting detecting change.",
            30, 0, Integer.MAX_VALUE);
	
	public IntOption widthOption = new IntOption("widowSize", 'b',
			"Size if the window", 500, 1, Integer.MAX_VALUE);
	
	public IntOption metricOption = new IntOption("metricOption", 'm',
			"1: PMAUC-PH, 2: EWAUC-PH, 3: WAUC-PH, 4: GMean-PH", 1, 1, 4);
	
	public FloatOption driftLevelOption = new FloatOption("driftLevelOption", 'c',
			"Lambda parameter of the Page Hinkley Test", 50, 1, Float.MAX_VALUE);
	
	public FloatOption warningLevelOption = new FloatOption("warningLevelOption", 'w',
			"Smaller lambda parameter of the Page Hinkley Test for WARNING alarm", 30, 1, Float.MAX_VALUE);
	
    public FloatOption deltaOption = new FloatOption("delta", 'd',
            "Delta parameter of the Page Hinkley Test", 0.005, 0.0, 1.0);
	
    public FloatOption alphaOption = new FloatOption("alpha", 'a',
            "Alpha parameter of the Page Hinkley Test", 1 - 0.0001, 0.0, 1.0);
	
    protected double Mint;
    protected double Maxt;
    protected double mt;
    protected long nt;
    protected double mean;
    protected double p;
	
	AUC_mClass auc_estimator;
	
	@Override
    public void resetLearning() {
        super.resetLearning();
        this.auc_estimator = null;
        
        this.mt = 0;
        this.Mint = Double.MAX_VALUE;
        this.Maxt = 0.0;
        this.nt = 1;
        this.mean = 0;
        this.p = 1;
    }
	
	@Override
	public void input(double prediction) {
		System.err.println("Error: PMAUC_WAUC_EWAUC_GMean_PH should not call input(prediction), only input(classVotes, inst).");
	}
	
    public void input(double[] classVotes, Instance inst) {
		if (this.isChangeDetected == true || this.isInitialized == false) {
            resetLearning();
            this.isInitialized = true;
        }
		
		if (this.auc_estimator == null) {
			auc_estimator = new AUC_mClass(this.widthOption.getValue(), inst.dataset().numClasses());
		}
		
		this.updatePAUCestimator(classVotes, inst);
		
		double x = 1;
		switch (this.metricOption.getValue()) {
			case 1:
				x = this.auc_estimator.getPMAUC();
				break;
			case 2:
				x = this.auc_estimator.getEqualWeightedAUC();
				break;
			case 3:
				x = this.auc_estimator.getWeightedAUC();
				break;
			case 4:
				x = this.auc_estimator.getGmean();
				break;
		}
		
		this.mean = this.mean + (x - this.mean) / (double) nt;
		
    	//Note: use "+delta" for monitoring the drop of PAUC, use "-delta" for monitoring the increase of the error
		this.mt = this.alphaOption.getValue()*this.mt + (x - this.mean + this.deltaOption.getValue());
		
    	nt++;
    	
    	if (this.mt >= this.Maxt) { // Maxt is the maximum of mt
    		this.Maxt = this.mt;
    	}
    	double pht = Maxt - mt;
//    	System.out.println("pht: " + pht);
    	
        this.estimation = this.mean;
        this.isChangeDetected = false;
        this.isWarningZone = false;
        this.delay = 0;
    	
    	if (nt < this.minNumInstancesOption.getValue()) {
    		return;
    	}
    	if (pht > this.driftLevelOption.getValue()) {
    		this.isChangeDetected = true;
    	} else if (pht > this.warningLevelOption.getValue()) {
    		this.isWarningZone = true;
    	}
	}
    
    private void updatePAUCestimator(double[] classVotes, Instance inst) {
    	double weight = inst.weight();
		if (!inst.classIsMissing()) {
			int trueClass = (int) inst.classValue();
			
			if (weight > 0.0) {
				double[] normalizedVote = classVotes.clone();//get a deep copy of classVotes
		  		  if (normalizedVote.length == inst.dataset().numClasses()) {

			  		  //// normalize and add score
			  		  double voteSum = 0.0;
			  		  for (double vote : normalizedVote) {
			  			  voteSum += vote;
			  		  }
			  		  if(normalizedVote.length == inst.dataset().numClasses()) {
			  			  if (voteSum > 0.0) { // noramlise only when the sum of votes is > 0.0
			  				  Utils.normalize(normalizedVote, voteSum);
			  			  }
			  		  }
			  		  
			  		  for(int i = 0; i < normalizedVote.length; i++) {
			  			  if(Double.isNaN(normalizedVote[i]))
			  				  normalizedVote[i] = 0.0;
			  		  }
		  		  } else {
		  			  normalizedVote = new double[inst.dataset().numClasses()];
		  			  Arrays.fill(normalizedVote, 0);
		  		  }

		  		  this.auc_estimator.add(normalizedVote, trueClass, Utils.maxIndex(classVotes) == trueClass);
			}
		}
    }

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
		
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub
		
	}
}