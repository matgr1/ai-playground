package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.activation.ActivationFunction;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

public class MaxPoolingLayer<NeuronT extends Neuron> extends ActivatableLayer<NeuronT> {

    private final ConvolutionDimensions dimensions;

    private final NeuronState<NeuronT>[][][] neurons;

    protected MaxPoolingLayer(NeuronFactory<NeuronT> neuronFactory,
                              int inputWidth,
                              int inputHeight,
                              int kernelWidth,
                              int kernelHeight,
                              int strideX,
                              int strideY,
                              int depth,
                              ActivationFunction activationFunction,
                              double... activationFunctionParameters) {

        super(neuronFactory, activationFunction, activationFunctionParameters);

        this.dimensions = new ConvolutionDimensions(
                inputWidth,
                inputHeight,
                kernelWidth,
                kernelHeight,
                strideX,
                strideY,
                depth);

        this.neurons = ConvolutionDimensions.createHiddenNeuronArray(
                this.neuronFactory,
                this.dimensions.outputWidth,
                this.dimensions.outputHeight,
                this.dimensions.depth);
    }

    protected MaxPoolingLayer(MaxPoolingLayer<NeuronT> other) {

        super(other);

        this.dimensions = other.dimensions.deepClone();

        this.neurons = ConvolutionDimensions.deepCloneNeuronArray(
                other.neurons,
                this.dimensions.outputWidth,
                this.dimensions.outputHeight,
                this.dimensions.depth);
    }

    @Override
    protected MaxPoolingLayer<NeuronT> deepClone() {
        return new MaxPoolingLayer<>(this);
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
    SizedIterable<NeuronState<NeuronT>> outputWritableNeurons() {

        return new NeuronArrayIterable<>(dimensions.outputWidth, dimensions.outputHeight, dimensions.depth, neurons);
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        // NOTE: nothing to do here... no weights...
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

        // TODO: this could be way more efficient... would definitely benefit from GPGPU processing
        // TODO: would also benefit from an array representation of the previous layer neurons... currently accessing
        //       neurons will likely be absurdly slow...

        // TODO: handle NaNs

        if ((dimensions.paddingX != 0) || (dimensions.paddingY != 0)) {

            // TODO: handle this (see above)
            throw new NotImplementedException();
        }

        for (int planeIndex = 0; planeIndex < dimensions.depth; planeIndex++) {

            NeuronState<NeuronT>[][] outputPlane = neurons[planeIndex];

            for (int outputY = 0; outputY < dimensions.outputHeight; outputY++) {

                NeuronState<NeuronT>[] outputRow = outputPlane[outputY];

                for (int outputX = 0; outputX < dimensions.outputWidth; outputX++) {

                    NeuronState<NeuronT> outputNeuron = outputRow[outputX];
                    NeuronState<NeuronT> maxInputNeuron = getMaxInputNeuron(previousLayerNeurons, outputX, outputY);

                    outputNeuron.preSynapse = maxInputNeuron.postSynapse;

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
    }

    @Override
    void resetPostSynapseErrorDerivatives(double value) {

        for (int z = 0; z < this.dimensions.depth; z++) {

            NeuronState<NeuronT>[][] plane = neurons[z];

            for (int y = 0; y < dimensions.outputHeight; y++) {

                NeuronState<NeuronT>[] row = plane[y];

                for (int x = 0; x < dimensions.outputWidth; x++) {

                    NeuronState<NeuronT> neuron = row[x];
                    neuron.postSynapseErrorDerivative = value;
                }
            }
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias, double learningRate) {

        if ((dimensions.paddingX != 0) || (dimensions.paddingY != 0)) {

            // TODO: handle this (see above)
            throw new NotImplementedException();
        }

        for (int planeIndex = 0; planeIndex < dimensions.depth; planeIndex++) {

            NeuronState<NeuronT>[][] outputPlane = neurons[planeIndex];

            for (int outputY = 0; outputY < dimensions.outputHeight; outputY++) {

                NeuronState<NeuronT>[] outputRow = outputPlane[outputY];

                for (int outputX = 0; outputX < dimensions.outputWidth; outputX++) {

                    NeuronState<NeuronT> outputNeuron = outputRow[outputX];
                    NeuronState<NeuronT> maxInputNeuron = getMaxInputNeuron(previousLayerNeurons, outputX, outputY);

                    double dE_dOut = outputNeuron.postSynapseErrorDerivative;
                    double dOut_dIn = computePreSynapseOutputDerivative(outputNeuron);

                    double dE_dIn = dE_dOut * dOut_dIn;

                    // update previous neuron dE/dOut
                    // TODO: don't need to compute this on the last pass
                    double dIn_dOutPrev = 1.0; // NOTE: weights are all effectively "1"
                    double dE_dOutPrev = (dE_dIn * dIn_dOutPrev);

                    maxInputNeuron.postSynapseErrorDerivative = dE_dOutPrev;
                }
            }
        }
    }

    private NeuronState<NeuronT> getMaxInputNeuron(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons,
                                                   int outputX,
                                                   int outputY) {

        NeuronState<NeuronT> maxInputNeuron = null;

        for (int kernelY = 0; kernelY < dimensions.kernelHeight; kernelY++) {

            int inputY = kernelY + (outputY * dimensions.strideY);
            int inputRowIndex = inputY * dimensions.inputWidth;

            for (int kernelX = 0; kernelX < dimensions.kernelWidth; kernelX++) {

                int inputX = kernelX + (outputX * dimensions.strideX);
                int inputIndex = inputRowIndex + inputX;

                NeuronState<NeuronT> inputNeuron = previousLayerNeurons.get(inputIndex);

                if (maxInputNeuron == null) {

                    maxInputNeuron = inputNeuron;

                } else {

                    if (inputNeuron.postSynapse > maxInputNeuron.postSynapse) {

                        maxInputNeuron = inputNeuron;
                    }
                }
            }
        }

        return maxInputNeuron;
    }
}
