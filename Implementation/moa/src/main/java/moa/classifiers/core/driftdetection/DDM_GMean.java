/*
 *    DDM_GMean.java
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
 * 	DDM based on G-Mean.
 *
 *  @author Chun Wai Chiu (cxc1015@student.bham.ac.uk / michaelchiucw@gmail.com)
 *  @version $Revision: 1 $
 */
public class DDM_GMean extends DDM {
	
	public FloatOption fadingFactorOption = new FloatOption("fadingFactor", 'f',
			"Fading factor for the time-decayed G-Mean", 0.999, 0, 1);
	
	protected double[] recallEstimation;
	protected double[] recallb;
	
	protected double timeDecayed_m_n;
	
	protected double n;
	
	protected double m_psmax;

    protected double m_pmax;

    protected double m_smax;

	public DDM_GMean() {
		super();
	}
	
	@Override
	public void resetLearning() {
		super.resetLearning();
		this.recallEstimation = null;
		this.recallb = null;
		
		this.timeDecayed_m_n = 0;
		this.n = 0;
		
		m_psmax = Double.MIN_VALUE;
		m_pmax = Double.MIN_VALUE;
		m_smax = Double.MIN_VALUE;
	}
	
	@Override
	public void input(double prediction) {
		System.err.println("Error: DDM_GMean should not call input(prediction), only input(prediction, actual).");
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
         * 
         * Initialising recallEstimation, recallb to contain zero-s.
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
		
        this.updatePrequentialGMean(prediction, actual);
        
        m_p = this.getPrequentialGMean();
        timeDecayed_m_n = this.fadingFactorOption.getValue() * timeDecayed_m_n + 1.0;
        m_s = Math.sqrt(m_p * (1 - m_p) / timeDecayed_m_n);
        
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
	
	protected double[] createDoubleArrayZeros(int numClasses) {
		double[] to_return = new double[numClasses];
		for (int i = 0; i < to_return.length; ++i) {
			to_return[i] = 0d;
		}
		return to_return;
	}
	
	protected void updatePrequentialGMean(double prediction, double actual) {
		int trueClass = (int) actual;
		
		/**
		 *  prediction == 0.0 means correct prediction; prediction == 1.0 means incorrect prediction
		 *  i.e. (prediction == 0.0 ? 1.0 : 0.0) is a translation for tracking the recall. Otherwise, it will be the "error of the recall".
		 */
		this.recallEstimation[trueClass] = this.fadingFactorOption.getValue() * this.recallEstimation[trueClass] + (prediction == 0.0 ? 1.0 : 0.0);
		this.recallb[trueClass] = this.fadingFactorOption.getValue() * this.recallb[trueClass] + 1.0;
	}
	
	protected double getRecallStatistic(int numClass) {
		return recallb[numClass] > 0.0? recallEstimation[numClass] / recallb[numClass] : 0.0;
	}
	
	protected double getPrequentialGMean() {
		double gmean = 0.0;
		
		for (int i = 0; i < this.recallEstimation.length; ++i) {
			if (i == 0) {
				gmean = this.getRecallStatistic(i);
			} else {
				gmean *= this.getRecallStatistic(i);
			}
		}
		gmean = Math.pow(gmean, (1.0/this.recallEstimation.length));
		
		return gmean;
	}
}
