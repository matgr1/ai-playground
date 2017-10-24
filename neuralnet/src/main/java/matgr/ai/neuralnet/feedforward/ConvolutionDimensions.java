package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;

import java.lang.reflect.Array;

public class ConvolutionDimensions {

    public final int inputWidth;
    public final int inputHeight;
    public final int inputDepth;

    public final int kernelWidth;
    public final int kernelHeight;
    public final int kernelDepth;

    public final int strideX;
    public final int strideY;
    public final int strideZ;

    public final int outputWidth;
    public final int outputHeight;
    public final int outputDepth;

    public final int paddingX;
    public final int paddingY;
    public final int paddingZ;

    public ConvolutionDimensions(int inputWidth,
                                 int inputHeight,
                                 int inputDepth,
                                 int kernelWidth,
                                 int kernelHeight,
                                 int kernelDepth) {

        this(
                inputWidth,
                inputHeight,
                inputDepth,
                kernelWidth,
                kernelHeight,
                kernelDepth,
                1,
                1,
                1);
    }

    public ConvolutionDimensions(int inputWidth,
                                 int inputHeight,
                                 int inputDepth,
                                 int kernelWidth,
                                 int kernelHeight,
                                 int kernelDepth,
                                 int strideX,
                                 int strideY,
                                 int strideZ) {

        this(
                inputWidth,
                inputHeight,
                inputDepth,
                kernelWidth,
                kernelHeight,
                kernelDepth,
                strideX,
                strideY,
                strideZ,
                0,
                0,
                0);
    }

    private ConvolutionDimensions(int inputWidth,
                                  int inputHeight,
                                  int inputDepth,
                                  int kernelWidth,
                                  int kernelHeight,
                                  int kernelDepth,
                                  int strideX,
                                  int strideY,
                                  int strideZ,
                                  int paddingX,
                                  int paddingY,
                                  int paddingZ) {

        // TODO: pretty sure larger strides don't make sense... (there will be gaps... if gaps end up being
        //       valid, the convolutions should center the kernels?)
        if (strideX > kernelWidth) {
            throw new IllegalArgumentException("Invalid horizontal stride");
        }
        if (strideY > kernelHeight) {
            throw new IllegalArgumentException("Invalid vertical stride");
        }
        if (strideZ > kernelDepth) {
            throw new IllegalArgumentException("Invalid outputDepth stride");
        }

        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.kernelDepth = kernelDepth;

        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;

        this.strideX = strideX;
        this.strideY = strideY;
        this.strideZ = strideZ;

        // TODO: automatically pad where necessary?
        this.paddingX = paddingX;
        this.paddingY = paddingY;
        this.paddingZ = paddingZ;

        this.outputWidth = 1 + (this.inputWidth - this.kernelWidth + (2 * paddingX)) / strideX;
        this.outputHeight = 1 + (this.inputHeight - this.kernelHeight + (2 * paddingY)) / strideY;
        this.outputDepth = 1 + (this.inputDepth - this.kernelDepth + (2 * paddingZ)) / strideZ;

        if (this.outputWidth < 0) {
            throw new IllegalArgumentException("Invalid kernel outputWidth");
        }
        if (this.outputHeight < 0) {
            throw new IllegalArgumentException("Invalid kernel outputHeight");
        }
        if (this.outputDepth < 0) {
            throw new IllegalArgumentException("Invalid kernel outputDepth");
        }
    }

    public ConvolutionDimensions deepClone() {

        return new ConvolutionDimensions(
                inputWidth,
                inputHeight,
                inputDepth,
                kernelWidth,
                kernelHeight,
                kernelDepth,
                strideX,
                strideY,
                strideZ,
                paddingX,
                paddingY,
                paddingZ);
    }

    public int inputCount() {
        return inputWidth * inputHeight * inputDepth;
    }

    public int outputCount() {
        return outputWidth * outputHeight * outputDepth;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][][] createHiddenNeuronArray(
            NeuronFactory<NeuronT> neuronFactory,
            int width,
            int height,
            int depth) {

        NeuronState<NeuronT>[][][] neurons = createEmptyNeuronArray(width, height, depth);

        for (int z = 0; z < depth; z++) {

            NeuronState<NeuronT>[][] plane = neurons[z];

            for (int y = 0; y < height; y++) {

                NeuronState<NeuronT>[] row = plane[y];

                for (int x = 0; x < width; x++) {

                    NeuronT neuron = neuronFactory.createHidden();
                    row[x] = new NeuronState<>(neuron);
                }
            }
        }

        return neurons;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][][] deepCloneNeuronArray(
            NeuronState<NeuronT>[][][] neurons,
            int width,
            int height,
            int depth) {

        NeuronState<NeuronT>[][][] neuronsClone = createEmptyNeuronArray(width, height, depth);

        for (int z = 0; z < depth; z++) {

            NeuronState<NeuronT>[][] sourcePlane = neurons[z];
            NeuronState<NeuronT>[][] targetPlane = neuronsClone[z];

            for (int y = 0; y < height; y++) {

                NeuronState<NeuronT>[] sourceRow = sourcePlane[y];
                NeuronState<NeuronT>[] targetRow = targetPlane[y];

                for (int x = 0; x < width; x++) {

                    targetRow[x] = sourceRow[x].deepClone();
                }
            }
        }

        return neurons;
    }

    public static <NeuronT extends Neuron> NeuronState<NeuronT>[][][] createEmptyNeuronArray(
            int width,
            int height,
            int depth) {

        @SuppressWarnings("unchecked")
        NeuronState<NeuronT>[][][] neurons = (NeuronState<NeuronT>[][][]) Array.newInstance(
                NeuronState.class,
                depth,
                height,
                width);

        return neurons;
    }
}
