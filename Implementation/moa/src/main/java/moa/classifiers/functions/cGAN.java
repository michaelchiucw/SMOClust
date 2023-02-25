package moa.classifiers.functions;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.BaseTrainingListener;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.config.Sgd;

import java.util.List;
import java.util.Random;
import java.util.function.Supplier;


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
public class cGAN {
    private static final IUpdater UPDATER_ZERO = new NoOp();
    private static final int K = 5;

    public interface ComputationGraphProvider {
    	ComputationGraph provide(IUpdater updater);
    }

    protected Supplier<ComputationGraph> generatorSupplier;
    protected ComputationGraphProvider discriminatorSupplier;
    protected ComputationGraphProvider cganSupplier;

    protected ComputationGraph generator;
    protected ComputationGraph discriminator;
    protected ComputationGraph cgan;
    protected int latentDim;

    protected IUpdater updater;
    protected IUpdater biasUpdater;
    protected OptimizationAlgorithm optimizer;
    protected GradientNormalization gradientNormalizer;
    protected double gradientNormalizationThreshold;
    protected WorkspaceMode trainingWorkSpaceMode;
    protected WorkspaceMode inferenceWorkspaceMode;
    protected CacheMode cacheMode;
    protected long seed;
    
    protected Random random;
    
    protected int counterK;

    public cGAN(Builder builder) {
        this.generatorSupplier = builder.generator;
        this.discriminatorSupplier = builder.discriminator;
        this.cganSupplier = builder.cgan;
        this.latentDim = builder.latentDimension;
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
        
        initCGan();
    }

    public ComputationGraph getGenerator() {
        return generator;
    }

    public ComputationGraph getDiscriminator() {
        return discriminator;
    }
    
    public ComputationGraph getcGAN() {
        return cgan;
    }

    public Evaluation evaluateGan(DataSetIterator data) {
        return cgan.evaluate(data);
    }

    public Evaluation evaluateGan(DataSetIterator data, List<String> labelsList) {
        return cgan.evaluate(data, labelsList);
    }

    public void setGeneratorListeners(BaseTrainingListener[] listeners) {
        generator.setListeners(listeners);
    }

    public void setDiscriminatorListeners(BaseTrainingListener[] listeners) {
        discriminator.setListeners(listeners);
    }

    public void setGanListeners(BaseTrainingListener[] listeners) {
        cgan.setListeners(listeners);
    }


    // Assuming the content of classSizes array is sum to 1.0
    public void fit(DataSet next) {
        INDArray realData = next.getFeatures();
        int batchSize = (int) realData.shape()[0];
        
        INDArray classLabels = next.getLabels();
        int numClasses = (int) classLabels.shape()[1];
        double[] classSizes = new double[numClasses];
        
        // Normalise the class size
        for (int i = 0; i < classSizes.length; ++i) {
        	classSizes[i] = classLabels.getColumn(i).mean(0).getDouble(0);
        }
        
        // Add the original class labels as one of the input
//        INDArray originalLabels = Nd4j.create(flattenOriginalLabels,new int[]{batchSize,1});
        INDArray[] realDataWithY = new INDArray[] {realData, classLabels};

        /**
         * Sample from latent space and let the generator creates fake data.
         */
        INDArray randomLatentDataX = Nd4j.randn(new int[]{batchSize, this.latentDim});
        // Sample an array of fake Y, following the original imbalance ration.
        INDArray randomClassLabels = this.generateRandomClassLabels(batchSize, classSizes);
        // Ask the generator to produce fake data
        INDArray fakeData = generator.outputSingle(randomLatentDataX, randomClassLabels);
        // Add back the randomClassLabels as one of the input attributes to train Discriminator
        INDArray[] fakeDataWithY = new INDArray[] {fakeData, randomClassLabels};
        
        // Real data are marked as "1", fake data at "-1".
        /**
         * Wasserstein loss function:
         * When using in a discriminator, use a label of 1 for real and -1 for generated
         * instead of the 1 and 0 used in normal GANs.
         */
        float[] minusOnes = new float[batchSize];
		for (int i = 0; i < minusOnes.length; ++i) {
			minusOnes[i] = (float) -1;
		}
        MultiDataSet realSet = new MultiDataSet(realDataWithY, new INDArray[]{Nd4j.ones(batchSize, 1)});
        MultiDataSet fakeSet = new MultiDataSet(fakeDataWithY, new INDArray[]{Nd4j.create(minusOnes,new int[]{batchSize, 1})});
//        MultiDataSet fakeSet = new MultiDataSet(fakeDataWithY, new INDArray[]{Nd4j.zeros(batchSize, 1)});
//        MultiDataSet realSet = new MultiDataSet(realDataWithY, new INDArray[]{Nd4j.randn(0, 0.1, new long[] {batchSize, 1}, Nd4j.getRandom())});
//        MultiDataSet fakeSet = new MultiDataSet(fakeDataWithY, new INDArray[]{Nd4j.randn(1, 0.1, new long[] {batchSize, 1}, Nd4j.getRandom())});
        
//        for (int i = 0; i < 5; ++i) {
        	discriminator.fit(realSet);
        	discriminator.fit(fakeSet);
//        }
        	
//        if (this.counterK < K) {
//        	this.counterK++;
//        	return;
//        } else {
//        	this.counterK = 0;
//        }

        // Update the discriminator in the GAN network
        this.updateCGanWithDiscriminator();

        /**
         *  Generate a new set of adversarial examples and try to mislead the discriminator.
         *  by labeling the fake images as real data (as "1") we reward the generator when 
         *  it's output tricks the discriminator.
         */
        INDArray adversarialExamples = Nd4j.randn(new int[]{batchSize, latentDim});
        INDArray adversarialExamplesRamdomY = this.generateRandomClassLabels(batchSize, classSizes);
        INDArray[] adversarialExamplesWithY = new INDArray[] {adversarialExamples, adversarialExamplesRamdomY};
//
        MultiDataSet adversarialSet = new MultiDataSet(adversarialExamplesWithY, new INDArray[] {Nd4j.ones(batchSize, 1)});
//        MultiDataSet adversarialSet = new MultiDataSet(adversarialExamplesWithY, new INDArray[] {Nd4j.randn(0, 0.1, new long[] {batchSize, 1}, Nd4j.getRandom())});
        
        // Fit the cGAN on the adversarial set, trying to fool the discriminator by generating
        // better fake data.
        cgan.fit(adversarialSet);

        // Copy the GANs generator part to "generator".
        this.updateGeneratorFromCGan();
    }
    
    public INDArray generateRandomClassLabels(int numLabels, double[] classSizes) {
    	float[][] randY = new float[numLabels][classSizes.length];
        for (int i = 0; i < randY.length; ++i) {
        	double classSizeSum = 0d;
        	double rand = this.random.nextDouble();
        	for (int j = 0; j < randY[i].length; ++j) {
        		double classSizeSumOld = classSizeSum;
        		classSizeSum += classSizes[j];
        		randY[i][j] = (rand >= classSizeSumOld && rand < classSizeSum) ? 1f : 0f;
        	}
        }
        return Nd4j.create(randY);
    }

    private void initCGan() {
    	this.random = new Random(this.seed);
    	
        generator = generatorSupplier.get();
        generator.init();
        
        discriminator = discriminatorSupplier.provide(updater);
        discriminator.init();
        
        cgan = cganSupplier.provide(UPDATER_ZERO);
        cgan.init();
        
        this.updateCGanWithDiscriminator();
        this.updateGeneratorFromCGan();
    }

//    private void copyParamsToGan() {
//        int genLayerCount = generator.getLayers().length;
//        for (int i = 0; i < cgan.getLayers().length; i++) {
//            if (i < genLayerCount) {
//                generator.getLayer(i).setParams(cgan.getLayer(i).params());
//            } else {
//                discriminator.getLayer(i - genLayerCount).setParams(cgan.getLayer(i).params());
//            }
//        }
//    }

    /**
     * After the GAN has been trained on misleading images, we update the generator the
     * new weights (we don't have to update the discriminator, as it is frozen in the GAN).
     */
    private void updateGeneratorFromCGan() {
        for (int i = 0; i < generator.getLayers().length; i++) {
            generator.getLayer(i).setParams(cgan.getLayer(i).params());
        }
    }

    /**
     * After the discriminator has been trained, we update the respective parts of the GAN network
     * as well.
     */
    private void updateCGanWithDiscriminator() {
    	int genLayerCount = generator.getLayers().length;
        for (int i = genLayerCount; i < cgan.getLayers().length; i++) {
            cgan.getLayer(i).setParams(discriminator.getLayer(i - genLayerCount).params());
        }
    }

    /**
     * GAN builder, used as a starting point for creating a MultiLayerConfiguration or
     * ComputationGraphConfiguration.<br>
     */
    public static class Builder implements Cloneable {
        protected Supplier<ComputationGraph> generator;
        protected ComputationGraphProvider discriminator;
        protected ComputationGraphProvider cgan;
        protected int latentDimension;

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
         * Set the (fake) image generator of the GAN.
         *
         * @param generator MultilayerNetwork
         * @return Builder
         */
        public cGAN.Builder generator(Supplier<ComputationGraph> generator) {
            this.generator = generator;
            return this;
        }

        /**
         * Set the image discriminator of the GAN.
         *
         * @param discriminator MultilayerNetwork
         * @return Builder
         */
        public cGAN.Builder discriminator(ComputationGraphProvider discriminator) {
            this.discriminator = discriminator;
            return this;
        }
        
        /**
         * Set the whole cGAN
         * 
         * @param cgan ComputationGraph
         * @return Builder
         */
        public cGAN.Builder cgan(ComputationGraphProvider cgan) {
            this.cgan = cgan;
            return this;
        }

        /**
         * Set the latent dimension, i.e. the input vector space dimension of the generator.
         *
         * @param latentDimension latent space input dimension.
         * @return Builder
         */
        public cGAN.Builder latentDimension(int latentDimension) {
            this.latentDimension = latentDimension;
            return this;
        }


        /**
         * Random number generator seed. Used for reproducibility between runs
         */
        public cGAN.Builder seed(long seed) {
            this.seed = seed;
            Nd4j.getRandom().setSeed(seed);
            return this;
        }

        /**
         * Optimization algorithm to use. Most common: OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT
         *
         * @param optimizationAlgo Optimization algorithm to use when training
         */
        public cGAN.Builder optimizationAlgo(OptimizationAlgorithm optimizationAlgo) {
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
        public cGAN.Builder updater(IUpdater updater) {
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
        public cGAN.Builder biasUpdater(IUpdater updater) {
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
        public cGAN.Builder gradientNormalization(GradientNormalization gradientNormalization) {
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
        public cGAN.Builder gradientNormalizationThreshold(double threshold) {
            this.gradientNormalizationThreshold = threshold;
            return this;
        }

        public cGAN build() {
            return new cGAN(this);
        }

    }

}