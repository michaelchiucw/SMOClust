package moa.classifiers.core.diversitytest;

import java.util.ArrayList;
import java.util.List;

import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.tasks.TaskMonitor;

public class EntropyMeasure extends AbstractOptionHandler implements DiversityTest {
	
	private static final long serialVersionUID = 1L;
	
	private List<Instance> testChunk;
	private List<Classifier> targetPool;
	
	private boolean isSet;
	
	public EntropyMeasure() {
		this.testChunk = null;
		this.targetPool = null;
		this.isSet = false;
	}
	
	public EntropyMeasure(List<Instance> chunk, List<Classifier> pool) {
		this.testChunk = new ArrayList<Instance>(chunk);
		this.targetPool = new ArrayList<Classifier>(pool);
		this.isSet = true;
	}

	@Override
	public void getDescription(StringBuilder sb, int indent) {
		// TODO Auto-generated method stub
	}

	@Override
	public Double call() throws Exception {
		if (isSet) {
			return getEntropyMeasure();
		} else {
			return 0.0;
		}
	}
	
	protected Double getEntropyMeasure() {
		double entropySum = 0.0;
		for (Instance instance : testChunk) {
			double correctSum = 0.0;
			for (Classifier classifier : targetPool) {
				correctSum += classifier.correctlyClassifies(instance) ? 1.0 : 0.0;
			}
			entropySum += Math.min(correctSum, this.targetPool.size() - correctSum) / (this.targetPool.size() - Math.ceil((this.targetPool.size() - 2)));
		}
		return entropySum / this.testChunk.size();
	}

	@Override
	public void set(List<Instance> testChunk, List<Classifier> targetPool) {
		this.testChunk = new ArrayList<Instance>(testChunk);
		this.targetPool = new ArrayList<Classifier>(targetPool);
		this.isSet = true;
	}

	@Override
	protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
		// TODO Auto-generated method stub

	}

	@Override
	public boolean morePositiveMoreDiverse() {
		return true;
	}

}
