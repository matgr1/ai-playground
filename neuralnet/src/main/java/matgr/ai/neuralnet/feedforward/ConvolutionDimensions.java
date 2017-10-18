package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;

import java.lang.reflect.Array;

public class ConvolutionDimensions {

    public final int inputWidth;
    public final int inputHeight;

    public final int kernelWidth;
    public final int kernelHeight;

    public final int strideX;
    public final int strideY;

    public final int outputWidth;
    public final int outputHeight;

    public final int paddingX;
    public final int paddingY;

    public ConvolutionDimensions(int inputWidth,
                                 int inputHeight,
                                 int kernelWidth,
                                 int kernelHeight) {

        this(inputWidth, inputHeight, kernelWidth, kernelHeight, 1, 1);
    }

    public ConvolutionDimensions(int inputWidth,
                                 int inputHeight,
                                 int kernelWidth,
                                 int kernelHeight,
                                 int strideX,
                                 int strideY) {

        this(inputWidth, inputHeight, kernelWidth, kernelHeight, strideX, strideY, 0, 0);
    }

    private ConvolutionDimensions(int inputWidth,
                                  int inputHeight,
                                  int kernelWidth,
                                  int kernelHeight,
                                  int strideX,
                                  int strideY,
                                  int paddingX,
                                  int paddingY) {

        // TODO: pretty sure larger strides don't make sense... (there will be gaps... if gaps end up being
        //       valid, the convolutions should center the kernels?)
        if (strideX > kernelWidth) {
            throw new IllegalArgumentException("Invalid horizontal stride");
        }
        if (strideY > kernelHeight) {
            throw new IllegalArgumentException("Invalid vertical stride");
        }

        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;

        this.strideX = strideX;
        this.strideY = strideY;

        // TODO: automatically pad where necessary?
        this.paddingX = paddingX;
        this.paddingY = paddingY;

        this.outputWidth = 1 + (this.inputWidth - this.kernelWidth + (2 * paddingX)) / strideX;
        this.outputHeight = 1 + (this.inputHeight - this.kernelHeight + (2 * paddingY)) / strideY;

        if (this.outputWidth < 0) {
            throw new IllegalArgumentException("Invalid kernel width");
        }
        if (this.outputHeight < 0) {
            throw new IllegalArgumentException("Invalid kernel height");
        }
    }

    public ConvolutionDimensions deepClone() {

        return new ConvolutionDimensions(
                inputWidth,
                inputHeight,
                kernelWidth,
                kernelHeight,
                strideX,
                strideY,
                paddingX,
                paddingY);
    }

    public int inputCount() {
        return inputWidth * inputHeight;
    }

    public int outputCount() {
        return outputWidth * outputHeight;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][] createHiddenNeuronArray(
            NeuronFactory<NeuronT> neuronFactory,
            int width,
            int height) {

        NeuronState<NeuronT>[][] neurons = createEmptyNeuronArray(width, height);

        for (int y = 0; y < height; y++) {

            NeuronState<NeuronT>[] row = neurons[y];

            for (int x = 0; x < width; x++) {

                NeuronT neuron = neuronFactory.createHidden();
                row[x] = new NeuronState<>(neuron);
            }
        }

        return neurons;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][] deepCloneNeuronArray(
            NeuronState<NeuronT>[][] neurons,
            int width,
            int height) {

        NeuronState<NeuronT>[][] neuronsClone = createEmptyNeuronArray(width, height);

        for (int y = 0; y < height; y++) {

            NeuronState<NeuronT>[] sourceRow = neurons[y];
            NeuronState<NeuronT>[] targetRow = neuronsClone[y];

            for (int x = 0; x < width; x++) {

                targetRow[x] = sourceRow[x].deepClone();
            }
        }

        return neurons;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][] createEmptyNeuronArray(int width, int height) {

        @SuppressWarnings("unchecked")
        NeuronState<NeuronT>[][] neurons = (NeuronState<NeuronT>[][]) Array.newInstance(
                NeuronState.class,
                height,
                width);

        return neurons;
    }
}
