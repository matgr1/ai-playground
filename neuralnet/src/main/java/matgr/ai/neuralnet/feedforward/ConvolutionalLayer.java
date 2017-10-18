package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class ConvolutionalLayer<NeuronT extends Neuron> extends ActivatableLayer<NeuronT> {

    private final ConvolutionDimensions dimensions;

    private final NeuronState<NeuronT>[][] neurons;

    private double[][] weights;
    private double biasWeight;

    public ConvolutionalLayer(NeuronFactory<NeuronT> neuronFactory,
                              int inputWidth,
                              int inputHeight,
                              int kernelWidth,
                              int kernelHeight,
                              ActivationFunction activationFunction,
                              double... activationFunctionParameters) {

        super(neuronFactory, activationFunction, activationFunctionParameters);

        this.dimensions = new ConvolutionDimensions(inputWidth, inputHeight, kernelWidth, kernelHeight);

        this.neurons = ConvolutionDimensions.createHiddenNeuronArray(
                this.neuronFactory,
                this.dimensions.outputWidth,
                this.dimensions.outputHeight);

        this.weights = new double[this.dimensions.kernelHeight][this.dimensions.kernelWidth];
        this.biasWeight = 0.0;
    }

    protected ConvolutionalLayer(ConvolutionalLayer<NeuronT> other) {

        super(other);

        this.dimensions = other.dimensions.deepClone();

        this.neurons = ConvolutionDimensions.deepCloneNeuronArray(
                other.neurons,
                this.dimensions.outputWidth,
                this.dimensions.outputHeight);

        this.weights = new double[this.dimensions.kernelHeight][this.dimensions.kernelWidth];

        for (int y = 0; y < this.dimensions.kernelHeight; y++) {

            double[] sourceRow = other.weights[y];
            double[] targetRow = this.weights[y];

            System.arraycopy(sourceRow, 0, targetRow, 0, this.dimensions.kernelWidth);
        }
    }

    @Override
    public int inputCount() {
        return dimensions.inputCount();
    }

    @Override
    public int outputCount() {
        return dimensions.outputCount();
    }

    @Override
    public SizedIterable<NeuronT> outputNeurons() {

        return new SizedSelectIterable<>(outputWritableNeurons(), n -> n.neuron);
    }

    @Override
    protected SizedIterable<NeuronState<NeuronT>> outputWritableNeurons() {

        return new NeuronArrayIterable<>(dimensions.outputWidth, dimensions.outputHeight, neurons);
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        for (int y = 0; y < dimensions.kernelHeight; y++) {

            double[] row = weights[y];

            for (int x = 0; x < dimensions.kernelWidth; x++) {
                row[x] = getRandomWeight(random);
            }
        }

        biasWeight = getRandomWeight(random);
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        int expectedSize = inputCount();

        // TODO: if this takes a 2D representation, then this can adapt to new dimensions easier... might be able
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
        //       implemented below) ...could maybe just control this based on settings for input/output dimensions and
        //       the kernel size
        // TODO: need parameters for stride size (how much to shift filter at each step, below is stride size 1)

        // TODO: this could be way more efficient... would definitely benefit from GPGPU processing
        // TODO: would also benefit from an array representation of the previous layer neurons... currently accessing
        //       neurons will likely be absurdly slow...

        // TODO: handle NaNs

        // NOTE: this might technically be cross-correlation rather than convolution... does that really matter?
        //       it shouldn't as long as the derivatives are calculated correctly during back propagation?)

        if ((dimensions.strideX != 1) || (dimensions.strideY != 1) ||
                (dimensions.paddingX != 0) || (dimensions.paddingY != 0)) {

            // TODO: handle this (see above)
            throw new NotImplementedException();
        }

        for (int outputY = 0; outputY < dimensions.outputHeight; outputY++) {

            NeuronState<NeuronT>[] outputRow = neurons[outputY];

            for (int outputX = 0; outputX < dimensions.outputWidth; outputX++) {

                NeuronState<NeuronT> outputNeuron = outputRow[outputX];
                outputNeuron.preSynapse = 0.0;

                for (int kernelY = 0; kernelY < dimensions.kernelHeight; kernelY++) {

                    double[] weightsRow = weights[kernelY];

                    int inputY = kernelY + outputY;
                    int inputRowIndex = inputY * dimensions.inputWidth;

                    for (int kernelX = 0; kernelX < dimensions.kernelWidth; kernelX++) {

                        double weight = weightsRow[kernelX];

                        int inputX = kernelX + outputX;
                        int inputIndex = inputRowIndex + inputX;

                        NeuronState<NeuronT> inputNeuron = previousLayerNeurons.get(inputIndex);

                        outputNeuron.preSynapse += (weight * inputNeuron.postSynapse);

                        if (Double.isNaN(outputNeuron.preSynapse)) {
                            // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                            //       return a status code from this function)... it should be given
                            //       "input" and "weight" (so it can decide what to do based on input values being
                            //       infinite/NaN/etc... if it fails, then set this to 0.0
                            outputNeuron.preSynapse = 0.0;
                        }
                    }
                }

                outputNeuron.preSynapse += (biasWeight * bias);

                if (Double.isNaN(outputNeuron.preSynapse)) {
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                    //       return a status code from this function)... it should be given
                    //       "neuron.incomingBiasWeight" and "bias" (so it can decide what to do based on input values being
                    //       infinite/NaN/etc... if it fails, then set this to 0.0
                    outputNeuron.preSynapse = 0.0;
                }

                activateNeuron(outputNeuron);
            }
        }
    }

    @Override
    void resetPostSynapseErrorDerivatives(double value) {

        for (int y = 0; y < dimensions.outputHeight; y++) {

            NeuronState<NeuronT>[] row = neurons[y];

            for (int x = 0; x < dimensions.outputWidth; x++) {

                NeuronState<NeuronT> neuron = row[x];
                neuron.postSynapseErrorDerivative = value;
            }
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons,
                       double bias,
                       double learningRate) {

        if ((dimensions.strideX != 1) || (dimensions.strideY != 1) ||
                (dimensions.paddingX != 0) || (dimensions.paddingY != 0)) {

            // TODO: handle this (see above)
            throw new NotImplementedException();
        }

        double[][] newWeights = new double[dimensions.kernelHeight][dimensions.kernelWidth];

        for (int y = 0; y < dimensions.kernelHeight; y++) {

            double[] weightsRow = weights[y];
            double[] newWeightsRow = newWeights[y];

            System.arraycopy(weightsRow, 0, newWeightsRow, 0, dimensions.kernelWidth);
        }

        double newBiasWeight = biasWeight;

        for (int outputY = 0; outputY < dimensions.outputHeight; outputY++) {

            NeuronState<NeuronT>[] outputRow = neurons[outputY];

            for (int outputX = 0; outputX < dimensions.outputWidth; outputX++) {

                NeuronState<NeuronT> outputNeuron = outputRow[outputX];

                // TODO: share this with FullyConnectedLayer
                double dE_dOut = outputNeuron.postSynapseErrorDerivative;
                double dOut_dIn = computePreSynapseOutputDerivative(outputNeuron);

                double dE_dIn = dE_dOut * dOut_dIn;

                for (int kernelY = 0; kernelY < dimensions.kernelHeight; kernelY++) {

                    double[] weightsRow = weights[kernelY];
                    double[] newWeightsRow = newWeights[kernelY];

                    int inputY = kernelY + outputY;
                    int inputRowIndex = inputY * dimensions.inputWidth;

                    for (int kernelX = 0; kernelX < dimensions.kernelWidth; kernelX++) {

                        double currentWeight = weightsRow[kernelX];

                        int inputX = kernelX + outputX;
                        int inputIndex = inputRowIndex + inputX;

                        NeuronState<NeuronT> previousNeuron = previousLayerNeurons.get(inputIndex);

                        double dIn_dW = previousNeuron.postSynapse;
                        double dE_dW = dE_dIn * dIn_dW;

                        // update incoming connection weight
                        newWeightsRow[kernelX] -= (dE_dW * learningRate);

                        // update previous neuron dE/dOut
                        // TODO: don't need to compute this on the last pass
                        double dIn_dOutPrev = currentWeight;
                        double dE_dOutPrev = (dE_dIn * dIn_dOutPrev);
                        previousNeuron.postSynapseErrorDerivative += dE_dOutPrev;
                    }
                }

                // update incoming bias weight
                double dIn_dW_Bias = bias;
                double dE_dW_Bias = dE_dIn * dIn_dW_Bias;

                newBiasWeight -= (dE_dW_Bias * learningRate);
            }
        }

        weights = newWeights;
        biasWeight = newBiasWeight;
    }

    @Override
    protected ConvolutionalLayer<NeuronT> deepClone() {
        return new ConvolutionalLayer<>(this);
    }
}
