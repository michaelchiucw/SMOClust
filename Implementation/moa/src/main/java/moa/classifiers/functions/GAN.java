package moa.classifiers.functions;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.impl.ActivationLReLU;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;


/**
 * @author Chun Wai Chiu
 * 
 * Modified from the following implementation:
 * 
 * Implementation of vanilla Generative Adversarial Networks as introduced in https://arxiv.org/pdf/1406.2661.pdf.
 * <p>
 * A DL4J GAN is initialized from two networks: a generator and a discriminator and will build a third network,
 * the GAN network, from the first two.
 *
 * @author Max Pumperla
 */
public class GAN {
    protected MultiLayerNetwork generator;
    protected MultiLayerNetwork discriminator;
    protected MultiLayerNetwork gan;
    protected int latentDim;
    protected int numHiddenNodesG;
    protected int numHiddenNodesD;
    protected int numInstAttr;

    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected OptimizationAlgorithm optimizer;
    protected GradientNormalization gradientNormalizer;
    protected double gradientNormalizationThreshold;
    protected WorkspaceMode trainingWorkSpaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;
    protected CacheMode cacheMode;
    protected long seed;
    
    protected int counterK;

//    private Double[] discriminatorLearningRates;


    public GAN(Builder builder) {
        this.latentDim = builder.latentDimension;
        this.numHiddenNodesG = builder.numHiddenNodesG;
        this.numHiddenNodesD = builder.numHiddenNodesD;
        this.numInstAttr = builder.numInstAttr;
        this.updater = builder.iUpdater;
        this.biasUpdater = builder.biasUpdater;
        this.optimizer = builder.optimizationAlgo;
        this.gradientNormalizer = builder.gradientNormalization;
        this.gradientNormalizationThreshold = builder.gradientNormalizationThreshold;
        this.trainingWorkSpaceMode = builder.trainingWorkspaceMode;
        this.inferenceWorkspaceMode = builder.inferenceWorkspaceMode;
        this.cacheMode = builder.cacheMode;
        this.seed = builder.seed;
        
        this.counterK = 0;

        constructGAN();
    }

    public MultiLayerNetwork getGenerator() {
        return generator;
    }

    public MultiLayerNetwork getDiscriminator() {
        return discriminator;
    }
    
    public MultiLayerNetwork getGAN() {
        return gan;
    }

    public Evaluation evaluateGan(DataSetIterator data) {
        return gan.evaluate(data);
    }

    public Evaluation evaluateGan(DataSetIterator data, List<String> labelsList) {
        return gan.evaluate(data, labelsList);
    }


    public void setGeneratorListeners(BaseTrainingListener[] listeners) {
        generator.setListeners(listeners);
    }

    public void setDiscriminatorListeners(BaseTrainingListener[] listeners) {
        discriminator.setListeners(listeners);
    }

    public void setGanListeners(BaseTrainingListener[] listeners) {
        gan.setListeners(listeners);
    }

    public void fit(DataSetIterator realData, int numEpochs) {
        for (int i = 0; i < numEpochs; i++) {
//        	System.out.println("Epoch: " + i);
            while (realData.hasNext()) {
                // Get real images as features
                DataSet next = realData.next();
                fit(next);
            }
            realData.reset();
        }
    }

    public void fit(DataSet next) {
    	
        INDArray realImages = next.getFeatures();
//    	INDArray realImages = Nd4j.hstack(next.getFeatures(), next.getLabels()); // includes labels for training.
        int batchSize = (int) realImages.shape()[0];

        // Sample from latent space and let the generate create fake images.
        INDArray randomLatentData = Nd4j.randn(new int[]{batchSize, latentDim});
        INDArray fakeImages = generator.output(randomLatentData);

        // Real images are marked as "0", fake images at "1".
        /**
         * if Wasserstein loss function:
         * When using in a discriminator, use a label of 1 for real and -1 for generated
         * instead of the 1 and 0 used in normal GANs.
         * 
         * i.e. Real images are maked as "-1" while fake images at "1"
         * as the original GAN flipped the labels for training.
         */
        DataSet realSet = new DataSet(realImages, Nd4j.zeros(batchSize, 1));
        DataSet fakeSet = new DataSet(fakeImages, Nd4j.ones(batchSize, 1));
//        float[] minusOnes = new float[batchSize];
//		for (int i = 0; i < minusOnes.length; ++i) {
//			minusOnes[i] = (float) -1;
//		}
//        DataSet realSet = new DataSet(realImages, Nd4j.ones(batchSize, 1));
//        DataSet fakeSet = new DataSet(fakeImages, Nd4j.create(minusOnes,new int[]{batchSize, 1}));
        
//        DataSet realSet = new DataSet(realImages, Nd4j.create(minusOnes,new int[]{batchSize, 1}));
//        DataSet fakeSet = new DataSet(fakeImages, Nd4j.ones(batchSize, 1));
        
        // Fit the discriminator on a combined batch of real and fake images.
//        DataSet combined = DataSet.merge(Arrays.asList(realSet, fakeSet));

        discriminator.fit(realSet);
        discriminator.fit(fakeSet);

        // Update the discriminator in the GAN network
        updateGanWithDiscriminator();
        
        // Generate a new set of adversarial examples and try to mislead the discriminator.
        // by labeling the fake images as real images we reward the generator when it's output
        // tricks the discriminator.
        INDArray adversarialExamples = Nd4j.randn(new int[]{batchSize, latentDim});
//        INDArray misleadingLabels = Nd4j.ones(batchSize, 1);
        INDArray misleadingLabels = Nd4j.zeros(batchSize, 1);
//        INDArray misleadingLabels = Nd4j.create(minusOnes,new int[]{batchSize, 1});
        DataSet adversarialSet = new DataSet(adversarialExamples, misleadingLabels);

        // Set learning rate of discriminator part of gan to zero.
        /*for (int i = generator.getLayers().length; i < gan.getLayers().length; i++) {
            gan.setLearningRate(i, 0.0);
        }*/

        // Fit the GAN on the adversarial set, trying to fool the discriminator by generating
        // better fake images.
        gan.fit(adversarialSet);

        // Copy the GANs generator part to "generator".
        updateGeneratorFromGan();
    }
    
    private Layer[] genLayers() { // single hidden layer
    	return new Layer[] {
			new DenseLayer.Builder().nIn(this.latentDim).nOut(this.numHiddenNodesG).activation(new ActivationLReLU(0.2)).build(),
            new DenseLayer.Builder().nIn(this.numHiddenNodesG).nOut(this.numInstAttr).activation(Activation.SIGMOID).build()
    	};
    }
    
    private MultiLayerConfiguration gConf() {
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    			.seed(this.seed)
    			.updater(this.updater)
    			.weightInit(WeightInit.XAVIER)
    			.activation(Activation.IDENTITY)
    			.list(genLayers())
    			.build();
    	
    	return conf;
    }
    
    private Layer[] disLayers() { // single hidden layer
    	return new Layer[] {
    		new DenseLayer.Builder().nIn(this.numInstAttr).nOut(this.numHiddenNodesD).activation(new ActivationLReLU(0.2)).dropOut(1 - 0.3).build(),
            new OutputLayer.Builder(LossFunctions.LossFunction.XENT).nIn(this.numHiddenNodesD).nOut(1).activation(Activation.SIGMOID).build()
    	};
    }
    
    private MultiLayerConfiguration dConf() {
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(this.seed)
                .updater(this.updater)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.IDENTITY)
                .list(disLayers())
                .build();
    	
    	return conf;
    }
    
    private MultiLayerConfiguration ganConf() {
    	Layer[] gLayers = genLayers();
    	Layer[] dLayers = Arrays.stream(disLayers())
    			.map((layer) -> {
    				if (layer instanceof DenseLayer || layer instanceof OutputLayer) {
                        return new FrozenLayerWithBackprop(layer);
                    } else {
                        return layer;
                    }
    			}).toArray(Layer[]::new);
    	Layer[] ganLayers = ArrayUtils.addAll(gLayers, dLayers);
    	
    	MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
    			.seed(this.seed)
    			.updater(this.updater)
    			.weightInit(WeightInit.XAVIER)
    			.activation(Activation.IDENTITY)
    			.list(ganLayers)
    			.build();
    	
    	return conf;
    }
    
    private void constructGAN() {
    	this.generator = new MultiLayerNetwork(gConf());
    	this.discriminator = new MultiLayerNetwork(dConf());
    	this.gan = new MultiLayerNetwork(ganConf());
    	this.generator.init();
    	this.discriminator.init();
    	this.gan.init();
    	
    	this.copyParamsToGan();
    }

    private void copyParamsToGan() {
        int genLayerCount = generator.getLayers().length;
        for (int i = 0; i < gan.getLayers().length; i++) {
            if (i < genLayerCount) {
                generator.getLayer(i).setParams(gan.getLayer(i).params());
            } else {
                discriminator.getLayer(i - genLayerCount).setParams(gan.getLayer(i).params());
            }
        }
    }

    /**
     * After the GAN has been trained on misleading images, we update the generator the
     * new weights (we don't have to update the discriminator, as it is frozen in the GAN).
     */
    private void updateGeneratorFromGan() {
        for (int i = 0; i < generator.getLayers().length; i++) {
            generator.getLayer(i).setParams(gan.getLayer(i).params());
        }
    }

    /**
     * After the discriminator has been trained, we update the respective parts of the GAN network
     * as well.
     */
    private void updateGanWithDiscriminator() {
        int genLayerCount = generator.getLayers().length;
        for (int i = genLayerCount; i < gan.getLayers().length; i++) {
            gan.getLayer(i).setParams(discriminator.getLayer(i - genLayerCount).params());
        }
    }

    /**
     * GAN builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     */
    //TODO: builder
    public static class Builder implements Cloneable {
        protected int latentDimension;
        protected int numHiddenNodesG;
        protected int numHiddenNodesD;
        protected int numInstAttr;

        protected IUpdater iUpdater = new Sgd();
        protected IUpdater biasUpdater = null;
        protected long seed = System.currentTimeMillis();
        protected OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT;
        protected GradientNormalization gradientNormalization = GradientNormalization.None;
        protected double gradientNormalizationThreshold = 1.0;

        protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;
        protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;
        protected CacheMode cacheMode = CacheMode.NONE;


        public Builder() {
        }

        /**
         * Set the latent dimension, i.e. the input vector space dimension of the generator.
         *
         * @param latentDimension latent space input dimension.
         * @return Builder
         */
        public GAN.Builder latentDimension(int latentDimension) {
            this.latentDimension = latentDimension;
            return this;
        }
        
        public GAN.Builder hiddenNodesG(int numHiddenNodesG) {
        	this.numHiddenNodesG = numHiddenNodesG;
        	return this;
        }
        
        public GAN.Builder hiddenNodesD(int numHiddenNodesD) {
        	this.numHiddenNodesD = numHiddenNodesD;
        	return this;
        }
        
        public GAN.Builder instAttr(int numInstAttr) {
        	this.numInstAttr = numInstAttr;
        	return this;
        }


        /**
         * Random number generator seed. Used for reproducibility between runs
         */
        public GAN.Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public GAN.Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }


        /**
         * Gradient updater configuration. For example, {@link org.nd4j.linalg.learning.config.Adam}
         * or {@link org.nd4j.linalg.learning.config.Nesterovs}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use
         */
        public GAN.Builder updater(IUpdater updater) {
            this.iUpdater = updater;
            return this;
        }

        /**
         * Gradient updater configuration, for the biases only. If not set, biases will use the updater as
         * set by {@link #updater(IUpdater)}<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param updater Updater to use for bias parameters
         */
        public GAN.Builder biasUpdater(IUpdater updater) {
            this.biasUpdater = updater;
            return this;
        }

        /**
         * Gradient normalization strategy. Used to specify gradient renormalization, gradient clipping etc.
         * See {@link GradientNormalization} for details<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         *
         * @param gradientNormalization Type of normalization to use. Defaults to None.
         * @see GradientNormalization
         */
        public GAN.Builder gradientNormalization(GradientNormalization gradientNormalization) {
            this.gradientNormalization = gradientNormalization;
            return this;
        }

        /**
         * Threshold for gradient normalization, only used for GradientNormalization.ClipL2PerLayer,
         * GradientNormalization.ClipL2PerParamType, and GradientNormalization.ClipElementWiseAbsoluteValue<br>
         * Not used otherwise.<br>
         * L2 threshold for first two types of clipping, or absolute value threshold for last type of clipping.<br>
         * Note: values set by this method will be applied to all applicable layers in the network, unless a different
         * value is explicitly set on a given layer. In other words: values set via this method are used as the default
         * value, and can be overridden on a per-layer basis.
         */
        public GAN.Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        public GAN build() {
            return new GAN(this);
        }

    }

}