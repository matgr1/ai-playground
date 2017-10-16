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

    private final ConvolutionalLayerSizes sizes;

    private final NeuronState<NeuronT>[][] neurons;

    // TODO: bias weight?
    private final double[][] weights;

    public ConvolutionalLayer(NeuronFactory<NeuronT> neuronFactory,
                              int width,
                              int height,
                              int kernelRadiusX,
                              int kernelRadiusY,
                              ActivationFunction activationFunction,
                              double... activationFunctionParameters) {

        super(neuronFactory);

        this.sizes = new ConvolutionalLayerSizes(width, height, kernelRadiusX, kernelRadiusY);

        this.neurons = createNeuronArray(this.sizes.outputWidth, this.sizes.outputHeight);
        this.weights = new double[this.sizes.kernelHeight][this.sizes.kernelWidth];

        for (int y = 0; y < this.sizes.outputHeight; y++) {

            NeuronState<NeuronT>[] row = this.neurons[y];

            for (int x = 0; x < this.sizes.outputWidth; x++) {

                NeuronT neuron = neuronFactory.createHidden(activationFunction, activationFunctionParameters);
                row[x] = new NeuronState<>(neuron);
            }
        }
    }

    protected ConvolutionalLayer(ConvolutionalLayer<NeuronT> other) {

        super(other);

        this.sizes = other.sizes.deepClone();

        this.neurons = createNeuronArray(this.sizes.outputWidth, this.sizes.outputHeight);
        this.weights = new double[this.sizes.kernelHeight][this.sizes.kernelWidth];

        for (int y = 0; y < this.sizes.outputHeight; y++) {

            NeuronState<NeuronT>[] sourceRow = other.neurons[y];
            NeuronState<NeuronT>[] targetRow = this.neurons[y];

            for (int x = 0; x < this.sizes.outputWidth; x++) {

                NeuronState<NeuronT> sourceNeuron = sourceRow[x];
                targetRow[x] = sourceNeuron.deepClone();
            }
        }

        for (int y = 0; y < this.sizes.kernelHeight; y++) {

            double[] sourceRow = other.weights[y];
            double[] targetRow = this.weights[y];

            System.arraycopy(sourceRow, 0, targetRow, 0, this.sizes.kernelWidth);
        }
    }

    @Override
    public int neuronCount() {
        return this.sizes.outputWidth * this.sizes.outputHeight;
    }

    @Override
    public SizedIterable<NeuronT> neurons() {

        return new SizedSelectIterable<>(writableNeurons(), n -> n.neuron);
    }

    @Override
    protected SizedIterable<NeuronState<NeuronT>> writableNeurons() {

        return new SizedNeuronIterable(this.sizes.outputWidth, this.sizes.outputHeight, neurons);
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        for (int y = 0; y < this.sizes.kernelHeight; y++) {

            double[] row = this.weights[y];

            for (int x = 0; x < this.sizes.kernelWidth; x++) {
                row[x] = getRandomWeight(random);
            }
        }
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        int expectedSize = this.sizes.inputWidth * this.sizes.inputHeight;

        // TODO: if this takes a 2D representation, then this can adapt to new sizes easier... might be able
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

        // TODO: need parameters for wide vs narrow filter (wide means zero padding, narrow is what is currently
        //       implemented below) ...could maybe just control this based on settings for input/output sizes and
        //       the kernel size
        // TODO: need parameters for stride size (how much to shift filter at each step, below is stride size 1)

        // TODO: this could be way more efficient... would definitely benefit from GPGPU processing
        // TODO: would also benefit from an array representation of the previous layer neurons... currently accessing
        //       neurons will likely be absurdly slow...

        // TODO: handle NaNs

        // NOTE: this might technically be cross-correlation rather than convolution... does that really matter?
        //       it shouldn't as long as the derivatives are calculated correctly during back propagation?)

        for (int y = 0; y < this.sizes.outputHeight; y++) {

            NeuronState<NeuronT>[] targetRow = neurons[y];

            for (int x = 0; x < this.sizes.outputWidth; x++) {

                double sum = 0.0;

                for (int kernelY = 0; kernelY < this.sizes.kernelHeight; kernelY++) {

                    double[] weightsRow = weights[kernelY];

                    int inputY = kernelY + y;
                    int inputRowIndex = inputY * this.sizes.inputWidth;

                    for (int kernelX = 0; kernelX < this.sizes.kernelWidth; kernelX++) {

                        double weight = weightsRow[kernelX];

                        int inputX = kernelX + x;
                        int inputIndex = inputRowIndex + inputX;

                        NeuronState<NeuronT> inputNeuron = previousLayerNeurons.get(inputIndex);

                        sum += (weight * inputNeuron.postSynapse);
                    }
                }

                NeuronState<NeuronT> targetNeuron = targetRow[x];

                targetNeuron.preSynapse = sum;

                ActivationFunction activationFunction = targetNeuron.neuron.getActivationFunction();
                double[] activationFunctionParameters = targetNeuron.neuron.getActivationFunctionParameters();

                targetNeuron.postSynapse = activationFunction.compute(targetNeuron.preSynapse, activationFunctionParameters);
            }
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons,
                       double bias,
                       double learningRate) {
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
        private final int height;

        private final NeuronState<NeuronT>[][] neurons;

        public SizedNeuronIterable(int width, int height, NeuronState<NeuronT>[][] neurons) {
            this.width = width;
            this.height = height;
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
