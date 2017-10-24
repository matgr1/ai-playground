package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.function.Function;

class ConvolutionInstance<NeuronT extends Neuron> {

    public final int outputWidth;
    public final int outputHeight;
    public final int outputDepth;

    public final int kernelWidth;
    public final int kernelHeight;
    public final int kernelDepth;

    public final NeuronState<NeuronT>[][][] neurons;

    public double[][][] weights;
    public double biasWeight;

    public ConvolutionInstance(NeuronFactory<NeuronT> neuronFactory, ConvolutionDimensions dimensions) {

        this.outputWidth = dimensions.outputWidth;
        this.outputHeight = dimensions.outputHeight;
        this.outputDepth = dimensions.outputDepth;

        this.kernelWidth = dimensions.kernelWidth;
        this.kernelHeight = dimensions.kernelHeight;
        this.kernelDepth = dimensions.kernelDepth;

        this.neurons = ConvolutionDimensions.createHiddenNeuronArray(
                neuronFactory,
                dimensions.outputWidth,
                dimensions.outputHeight,
                dimensions.outputDepth);

        this.weights = new double
                [dimensions.kernelDepth]
                [dimensions.kernelHeight]
                [dimensions.kernelWidth];

        this.biasWeight = 0.0;
    }

    private ConvolutionInstance(ConvolutionInstance<NeuronT> other) {

        this.outputWidth = other.outputWidth;
        this.outputHeight = other.outputHeight;
        this.outputDepth = other.outputDepth;

        this.kernelWidth = other.kernelWidth;
        this.kernelHeight = other.kernelHeight;
        this.kernelDepth = other.kernelDepth;

        this.neurons = ConvolutionDimensions.deepCloneNeuronArray(
                other.neurons,
                this.outputWidth,
                this.outputHeight,
                this.outputDepth);

        this.weights = new double
                [this.kernelDepth]
                [this.kernelHeight]
                [this.kernelWidth];

        for (int z = 0; z < this.kernelDepth; z++) {

            double[][] sourcePlane = other.weights[z];
            double[][] targetPlane = this.weights[z];

            for (int y = 0; y < this.kernelHeight; y++) {

                double[] sourceRow = sourcePlane[y];
                double[] targetRow = targetPlane[y];

                System.arraycopy(sourceRow, 0, targetRow, 0, this.kernelWidth);
            }
        }
    }

    public ConvolutionInstance<NeuronT> deepClone() {

        return new ConvolutionInstance<>(this);
    }

    public void randomizeWeights(RandomGenerator random, Function<RandomGenerator, Double> getRandomWeight) {

        for (int z = 0; z < this.kernelDepth; z++) {

            double[][] plane = weights[z];

            for (int y = 0; y < kernelHeight; y++) {

                double[] row = plane[y];

                for (int x = 0; x < kernelWidth; x++) {
                    row[x] = getRandomWeight.apply(random);
                }
            }
        }

        // TODO: bias weight per plane?
        biasWeight = getRandomWeight.apply(random);
    }
}
