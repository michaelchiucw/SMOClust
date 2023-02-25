package moa.classifiers.functions;

import java.util.ArrayList;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import weka.classifiers.functions.dl4j.Utils;
import weka.core.Attribute;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;


public class testing extends AbstractClassifier implements MultiClassClassifier {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	public IntOption numHiddenNodesGOption = new IntOption("numHiddenNodesG", 'g',
			"numHiddenNodesG", 90, 1, Integer.MAX_VALUE);
	
	public IntOption numHiddenNodesDOption = new IntOption("numHiddenNodesD", 'd',
			"numHiddenNodesD", 45, 1, Integer.MAX_VALUE);
	
	public IntOption windowSizeOption = new IntOption("WindowSize", 'b',
			"WindowSize", 500, 1, Integer.MAX_VALUE);

	protected int windowSize;
	private Instances buffer;
	
	protected SamoaToWekaInstanceConverter instanceConverter;
	
	private boolean hasInitG;
	private MultiLayerNetwork generator;
	private int numHiddenNodesG;
	
	private boolean hasInitD;
	private MultiLayerNetwork discriminator;
	private int numHiddenNodesD;
	
	private boolean hasbuilt;
	
//	public GAN(int numHiddenNodesG, int numHiddenNodesD) {
//		this.numHiddenNodesG = numHiddenNodesG;
//		this.numHiddenNodesD = numHiddenNodesD;
//		
//		this.hasInitG = false;
//		this.hasInitD = false;
//		
//		
//	}
	
	@Override
	public void resetLearningImpl() {
		
		this.windowSize = this.windowSizeOption.getValue();
		
		this.numHiddenNodesG = this.numHiddenNodesGOption.getValue();
		this.numHiddenNodesD = this.numHiddenNodesDOption.getValue();
		
		this.instanceConverter = new SamoaToWekaInstanceConverter();
		
		this.hasInitG = false;
		this.hasInitD = false;
		
		this.hasbuilt = false;
		
	}
	
	private void initG(int inDim, int numHiddenNodes, int numOutputNodes) {
		
		Layer[] layers = new Layer[] {
				new DenseLayer.Builder().nIn(inDim).nOut(numHiddenNodes)
							  .activation(Activation.RELU)
							  .build(),
                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                			   .nIn(numHiddenNodes).nOut(numOutputNodes)
                               .weightInit(WeightInit.XAVIER)
                               .activation(Activation.SOFTMAX)
                               .build()
		};
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
																 .seed(1)
																 .weightInit(WeightInit.XAVIER)
																 .updater(new Adam())
																 .list(layers)
																 .build();
		this.generator = new MultiLayerNetwork(conf);
		this.generator.init();

	}
	
	private void initD(int inDim, int numHiddenNodes) {
	
		Layer[] layers = new Layer[] {
				new DenseLayer.Builder().nIn(inDim).nOut(numHiddenNodes)
							  .activation(Activation.RELU)
							  .build(),
                new OutputLayer.Builder(LossFunction.XENT)
                			   .nIn(numHiddenNodes).nOut(2)
                               .weightInit(WeightInit.XAVIER)
                               .activation(Activation.SIGMOID)
                               .build()
		};
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
																 .seed(1)
																 .weightInit(WeightInit.XAVIER)
																 .updater(new Adam())
																 .list(layers)
																 .build();
		this.discriminator = new MultiLayerNetwork(conf);
		this.discriminator.init();

	}
	
	public void trainGAN(Instances instSet) {
		// Convert original class attribute to be one of the dimension.
		// Add new class attribute -> real / fake
		
		weka.core.Instances wekaInstances = this.instanceConverter.wekaInstances(new Instances(instSet));
//		System.out.println(wekaInstances.toString());
		ArrayList<String> newClassAttrValues = new ArrayList<String>();
		newClassAttrValues.add("real");
		newClassAttrValues.add("fake");
		Attribute newClassAttr = new Attribute("isReal", newClassAttrValues);
		wekaInstances.insertAttributeAt(newClassAttr, wekaInstances.numAttributes());
		wekaInstances.setClassIndex(wekaInstances.numAttributes()-1);
		for (int i = 0; i < wekaInstances.size(); ++i) {
			wekaInstances.get(i).setClassValue("real");
		}
		
		System.out.println(wekaInstances.toString());
		
		if (!this.hasInitG) {
			this.initG(100, this.numHiddenNodesG, wekaInstances.numAttributes());
			this.hasInitG = true;
		}
		if (!this.hasInitD) {
//			System.out.println(instSet.toString());
			int inDimD = wekaInstances.numAttributes()-1;
			this.initD(inDimD, this.numHiddenNodesD);
			this.hasInitD = true;
		}
		
		try {
			if (!this.hasbuilt) {				
				DataSet dl4jDataSet = Utils.instancesToDataSet(wekaInstances);
				this.discriminator.fit(dl4jDataSet);
				this.hasbuilt = true;
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
//		System.out.println("G:\n" + this.generator.toString());
//		System.out.println("D:\n" + this.discriminator.toString());

	}
	
	private void trainD(Instance inst) {
		
	}
	
	private void trainG() {
		
	
	}

	@Override
	public boolean isRandomizable() {
		// TODO Auto-generated method stub
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return new double[]{0};
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
		
		if (this.buffer == null) {
			this.buffer = new Instances(inst.dataset(), this.windowSize);
		}
		if (this.buffer.numInstances() < this.windowSize) {
			this.buffer.add(inst);
		} else {
			this.trainGAN(buffer);
			this.buffer.delete();
		}
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
