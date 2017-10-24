package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;

class MaxPoolingInstance<NeuronT extends Neuron> {

    public final int outputWidth;
    public final int outputHeight;
    public final int outputDepth;

    public final int kernelWidth;
    public final int kernelHeight;
    public final int kernelDepth;

    public final NeuronState<NeuronT>[][][] neurons;

    public MaxPoolingInstance(NeuronFactory<NeuronT> neuronFactory, ConvolutionDimensions dimensions) {

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
    }

    private MaxPoolingInstance(MaxPoolingInstance<NeuronT> other) {

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
    }

    public MaxPoolingInstance<NeuronT> deepClone() {

        return new MaxPoolingInstance<>(this);
    }
}
