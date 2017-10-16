package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.NestedIterator;
import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import javax.annotation.Nonnull;
import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class ConvolutionalLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    private final int width;
    private final int height;
    private final int kernelWidth;
    private final int kernelHeight;

    private final int outputWidth;
    private final int outputHeight;

    private final NeuronState<NeuronT>[][] neurons;

    private final double[][] weights;

    public ConvolutionalLayer(NeuronFactory<NeuronT> neuronFactory,
                              int width,
                              int height,
                              int kernelWidth,
                              int kernelHeight,
                              ActivationFunction activationFunction,
                              double... activationFunctionParameters) {

        super(neuronFactory);

        if (kernelWidth > width) {
            throw new IllegalArgumentException("Kernel width cannot be larger than width");
        }
        if (kernelHeight > height) {
            throw new IllegalArgumentException("Kernel height cannot be larger than height");
        }

        this.width = width;
        this.height = height;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;

        this.outputWidth = 1 + (this.width - this.kernelWidth);
        this.outputHeight = 1 + (this.height - this.kernelHeight);

        this.neurons = createNeuronArray(this.outputWidth, this.outputHeight);
        this.weights = new double[this.kernelHeight][this.kernelWidth];

        for (int y = 0; y < this.outputHeight; y++) {

            NeuronState<NeuronT>[] row = this.neurons[y];

            for (int x = 0; x < this.outputWidth; x++) {

                NeuronT neuron = neuronFactory.createHidden(activationFunction, activationFunctionParameters);
                row[x] = new NeuronState<>(neuron);
            }
        }
    }

    protected ConvolutionalLayer(ConvolutionalLayer<NeuronT> other) {

        super(other);

        this.width = other.width;
        this.height = other.height;
        this.kernelWidth = other.kernelWidth;
        this.kernelHeight = other.kernelHeight;

        this.outputWidth = other.outputWidth;
        this.outputHeight = other.outputHeight;

        this.neurons = createNeuronArray(this.outputWidth, this.outputHeight);
        this.weights = new double[this.kernelHeight][this.kernelWidth];

        for (int y = 0; y < this.outputHeight; y++) {

            NeuronState<NeuronT>[] sourceRow = other.neurons[y];
            NeuronState<NeuronT>[] targetRow = this.neurons[y];

            for (int x = 0; x < this.outputWidth; x++) {

                NeuronState<NeuronT> sourceNeuron = sourceRow[x];
                targetRow[x] = sourceNeuron.deepClone();
            }
        }

        for (int y = 0; y < this.kernelHeight; y++) {

            double[] sourceRow = other.weights[y];
            double[] targetRow = this.weights[y];

            System.arraycopy(sourceRow, 0, targetRow, 0, this.kernelWidth);
        }
    }

    @Override
    public int neuronCount() {
        return outputWidth * outputHeight;
    }

    @Override
    public SizedIterable<NeuronT> neurons() {

        return new SizedSelectIterable<>(writableNeurons(), n -> n.neuron);
    }

    @Override
    protected SizedIterable<NeuronState<NeuronT>> writableNeurons() {

        return new SizedNeuronIterable(outputWidth, neurons);
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        for (int y = 0; y < this.kernelHeight; y++) {

            double[] row = this.weights[y];

            for (int x = 0; x < this.kernelWidth; x++) {
                row[x] = getRandomWeight(random);
            }
        }
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        int expectedSize = width * height;

        // TODO: if this passes a 2D representation, then this can adapt to new sizes easier... might be able
        //       to do more optimizations if it's 2D aware as well ...(a non-convolutional layer would then pass
        //       a 1xN sized set of neurons)
        // TODO: if above is true, then the constructor would only take kernel size and would do less initialization...
        // TODO: ...would still need to check kernel size here then...
        if (previousLayerNeurons.size() != expectedSize) {
            throw new IllegalStateException("Invalid previous layers size");
        }
    }

    @Override
    void activate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias) {
        throw new NotImplementedException();
    }

    @Override
    void backPropagate(double learningRate, double bias, SizedIterable<NeuronState<NeuronT>> previousLayerNeurons) {
        throw new NotImplementedException();
    }

    @Override
    protected ConvolutionalLayer<NeuronT> deepClone() {
        return new ConvolutionalLayer<>(this);
    }

    private NeuronState<NeuronT>[][] createNeuronArray(int width, int height) {

        @SuppressWarnings("unchecked")
        NeuronState<NeuronT>[][] value = (NeuronState<NeuronT>[][]) Array.newInstance(
                NeuronState.class,
                height,
                width);

        return value;
    }

    private class SizedNeuronIterable implements SizedIterable<NeuronState<NeuronT>> {

        private final int width;

        private final NeuronState<NeuronT>[][] neurons;

        public SizedNeuronIterable(int width, NeuronState<NeuronT>[][] neurons) {
            this.width = width;
            this.neurons = neurons;
        }

        @Override
        public int size() {
            return width * height;
        }

        @Override
        public NeuronState<NeuronT> get(int index) {

            int row = index / width;
            int col = index % width;

            return neurons[row][col];
        }

        @Override
        @Nonnull
        public Iterator<NeuronState<NeuronT>> iterator() {

            List<NeuronState<NeuronT>[]> neuronsList = Arrays.asList(neurons);
            return new NestedIterator<>(neuronsList, Arrays::asList);
        }
    }
}
