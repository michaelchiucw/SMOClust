package moa.classifiers.core.diversitytest;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import com.yahoo.labs.samoa.instances.Instance;

import moa.classifiers.Classifier;

public class DiversityParallelTester {
	private List<Instance> testChunck;
	private List<Classifier> classifierPool;
	private DiversityTest diversityTestType;

	public DiversityParallelTester(List<Instance> chunk, List<Classifier> pool, DiversityTest diversityTestType) {
		/**
		 * Copying instead of getting the reference to avoid affecting the actual pool and chunk.
		 */
		this.testChunck = new ArrayList<Instance>(chunk);
		this.classifierPool = new ArrayList<Classifier>(pool);
		this.diversityTestType = (DiversityTest) diversityTestType.copy();
	}
	
	public Classifier getSuggestedDeletion() {
		Classifier to_return = null;
		int index = performTest();
		if (index > -1) {
			to_return = this.classifierPool.get(index);
		}
		return to_return;
	}
	
	public int getSuggestedDeletionIndex() {
		return performTest();
	}
	
	private int performTest() {
		int to_return = -1;
		ExecutorService threadPool = Executors.newCachedThreadPool();
		Map<Integer, Future<Double>> results = new HashMap<>();
		for (int i = 0; i < this.classifierPool.size(); ++i) {
			List<Classifier> testingPool = new ArrayList<Classifier>(this.classifierPool);
			testingPool.remove(i);
			DiversityTest test = (DiversityTest) diversityTestType.copy();
			test.set(this.testChunck, testingPool);
			results.put(i, threadPool.submit(test));
		}
		int index = 0;
		double bestScore = 0;
		try {
			for (int i = 0 ; i < this.classifierPool.size(); ++i) {
				Future<Double> f = results.get(i);
				if (this.diversityTestType.morePositiveMoreDiverse()) {
					if (f.get() > bestScore) {
						index = i;
						bestScore = f.get();
					}
				} else {
					if (i == 0) {
						bestScore = f.get();
					} else if (f.get() < bestScore && i > 0) {
						index = i;
						bestScore = f.get();
					} else {
						/**
						 * Do nothing.
						 */
					}
				}
//				System.out.println("if remove the " + i + "-th classifier | marks: " + f.get() + " | is currently the best score? " + (bestScore == f.get() ? "Yes" : "No") + " | morePositiveMoreDiverse: " + this.diversityTestType.morePositiveMoreDiverse());
			}
		} catch (InterruptedException e) {
            System.out.println("Processing interrupted.");
        } catch (ExecutionException e) {
            throw new RuntimeException("Error computing diversity test.", e);
        }
		threadPool.shutdownNow();
//		if (bestScore > currentScore) {
		to_return = index;
//		System.out.println("Remove suggestion index: " + to_return);
//		}
		return to_return;
	}

}