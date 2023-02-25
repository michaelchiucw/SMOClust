/*
 *    DDM_OCI.java
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

import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;

/**
 * 	DDM based on Minority Class recall from Wang et al. IJCNN 2013.
 * 	<strong>Only works for binary classification problems.</strong>
 *
 *  <p>WANG, S.; MINKU, L.; GHEZZI, D.; CALTABIANO, D.; TINO, P.; YAO, X.; 
 *  "Concept Drift Detection for Online Class Imbalance Learning",
 *  Proceedings of the 2013 International Joint Conference on Neural Networks (IJCNN),
 *  10 pages, August 2013, doi: 10.1109/IJCNN.2013.6706768.</p>
 *
 *  @author Chun Wai Chiu (cxc1015@student.bham.ac.uk / michaelchiucw@gmail.com)
 *  @version $Revision: 1 $
 */
public class DDM_OCI extends DDM_GMean {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7412975496832664411L;
	
	public FloatOption thetaOption = new FloatOption("theta", 't',
            "The time decay factor for class size.", 0.99, 0, 1);
	
	protected double[] classSizeEstimation;
	protected double[] classSizeb;

	public DDM_OCI() {
		super();
	}
	
	@Override
	public void resetLearning() {
		super.resetLearning();
//		this.m_pArr = null;
//		this.examplesSeenByClass = null;
		this.classSizeEstimation = null;
		this.classSizeb = null;
	}
	
	@Override
	public void input(double prediction) {
		System.err.println("Error: DDM_OCI should not call input(prediction), only input(prediction, actual).");
	}
	
	// As with DDM, prediction is 0 for correct prediction and 1 for mistaken prediction
	// Actual is the actual class of the example being predicted
	public void input(double prediction, Instance inst) {
        // prediction must be 1 or 0
        // It monitors the g-mean
        if (this.isChangeDetected == true || this.isInitialized == false) {
            resetLearning();
            this.isInitialized = true;
        }
        
        double actual = inst.classValue();
        
        /**
         * recallEstimation, recallb are from the superclass, DDM_GMean
         * They are used to calculate time decay G-Mean
         * 
         * Initialising them to contain zero-s.
         * 
         * The original m_n initialised to m_n for the calculation at the
         * first time step that's why the increment it after the calculations
         * for the next time step. Otherwise, the way the calculation the m_p
         * at the first time step would be incorrect, because the equation is
         * actually using the old m_p times m_n-1 to get the old sum then use the
         * result for the new probability (new Bernoulli mean) calculation.
         * If m_n is initialised as 0, the m_n-1 at the first time step would be -1.
         * In the case of the calculation in this class, we are calculating the 
         * recalls prequentially. So, it is fine to initialise them as 0-s.
         */
        if (this.recallEstimation == null) {
			this.recallEstimation = this.createDoubleArrayZeros(inst.numClasses());
		}
		if (this.recallb == null) {
			this.recallb = this.createDoubleArrayZeros(inst.numClasses());
		}

        this.updateClassSize(inst);
        this.updatePrequentialGMean(prediction, actual);
   
        int minClassIndex = this.getMinorityClass();
        m_p = getRecallStatistic(minClassIndex);
        m_s = Math.sqrt(m_p * (1 - m_p) / recallb[minClassIndex]);
        timeDecayed_m_n = recallb[minClassIndex]; // b++ has been done in updatePrequentialGMean(), with time-decayed factor
        
        n++;
        
        // System.out.print(prediction + " " + m_n + " " + (m_p+m_s) + " ");
        this.estimation = m_p;
        this.isChangeDetected = false;
        this.isWarningZone = false;
        this.delay = 0;

        if (n <= minNumInstances) {
            return;
        }

        if (m_p + m_s >= m_psmax) {
        	m_pmax = m_p;
        	m_smax = m_s;
        	m_psmax = m_p + m_s;
        }

        if (n > minNumInstances && m_p - m_s <= m_pmax - outcontrolLevel * m_smax) {
            //System.out.println(m_p + ",D");
            this.isChangeDetected = true;
            //resetLearning();
        } else if (m_p - m_s <= m_pmax - warningLevel * m_smax) {
            //System.out.println(m_p + ",W");
            this.isWarningZone = true;
        } else {
            this.isWarningZone = false;
            //System.out.println(m_p + ",N");
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
		if (this.classSizeb == null) {
			classSizeb = new double[inst.numClasses()];
			
			for (int i=0; i<classSizeEstimation.length; ++i) {
				classSizeb[i] = 1d/classSizeb.length;
			}
		}
		
		for (int i=0; i<classSizeEstimation.length; ++i) {
			classSizeEstimation[i] = thetaOption.getValue() * classSizeEstimation[i] + ((int) inst.classValue() == i ? 1d:0d);
			classSizeb[i] = thetaOption.getValue() * classSizeb[i] + 1d;
		}
	}
	
	protected double getClassSize(int classIndex) {
		return classSizeb[classIndex] > 0.0 ? classSizeEstimation[classIndex] / classSizeb[classIndex] : 0.0;
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
