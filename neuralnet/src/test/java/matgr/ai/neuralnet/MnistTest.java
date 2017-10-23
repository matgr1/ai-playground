package matgr.ai.neuralnet;

import com.google.common.primitives.Doubles;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import matgr.ai.neuralnet.feedforward.ConvolutionDimensions;
import matgr.ai.neuralnet.feedforward.FeedForwardNeuralNet;

import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;


/**
 * Unit test for simple App.
 */
public class MnistTest extends TestCase {

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public MnistTest(String testName) {

        super(testName);
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(NeuralNetTest.class);
    }

    public void testMnist() throws IOException {

        MnistIdxFile trainingLabels = loadMnistIdxResourceFile("/train-labels.idx1-ubyte");
        MnistIdxFile trainingImages = loadMnistIdxResourceFile("/train-images.idx3-ubyte");

        List<TrainingSet> trainingSets = loadTrainingSet(trainingLabels, trainingImages);

        final int maxPoolingKernelWidth = 2;
        final int maxPoolingKernelHeight = 2;
        final int maxPoolingKernelStrideX = 2;
        final int maxPoolingKernelStrideY = 2;

        final int outputCount = 10;

        final double bias = 1;

        final double learningRate = 0.01;
        final int maxSteps = 1000000;
        final double maxErrorRms = 0.1;

        final int noProgressResetThreshold = 5000;

        final int inputCount = trainingImages.itemWidth * trainingImages.itemHeight;

        ActivationFunction convolutionalActivationFunction = KnownActivationFunctions.RELU;
        double[] convolutionalActivationFunctionParameters = convolutionalActivationFunction.defaultParameters();

        ActivationFunction maxPoolingActivationFunction = KnownActivationFunctions.IDENTITY;
        double[] maxPoolingActivationFunctionParameters = convolutionalActivationFunction.defaultParameters();

        final int hiddenNeuronCount = 10;
        ActivationFunction hiddenActivationFunction = KnownActivationFunctions.TANH;
        double[] hiddenActivationFunctionParameters = convolutionalActivationFunction.defaultParameters();

        // TODO: softmax requires IDENTITY activation function? enforce this if true?
        final boolean outputApplySoftmax = true;
        ActivationFunction outputActivationFunction = KnownActivationFunctions.IDENTITY;
        double[] outputActivationFunctionParameters = outputActivationFunction.defaultParameters();

        FeedForwardNeuralNet<Neuron> neuralNet = new FeedForwardNeuralNet<>(
                new DefaultNeuronFactory(),
                inputCount,
                outputCount,
                outputApplySoftmax,
                outputActivationFunction,
                outputActivationFunctionParameters);

        ConvolutionDimensions firstConvolutionDim = new ConvolutionDimensions(
                trainingImages.itemWidth,
                trainingImages.itemHeight,
                5,
                5,
                8);

        neuralNet.addConvolutionalHiddenLayer(
                firstConvolutionDim.inputWidth,
                firstConvolutionDim.inputHeight,
                firstConvolutionDim.kernelWidth,
                firstConvolutionDim.kernelHeight,
                firstConvolutionDim.depth,
                convolutionalActivationFunction,
                convolutionalActivationFunctionParameters);

        neuralNet.addMaxPoolingHiddenLayer(
                firstConvolutionDim.outputWidth,
                firstConvolutionDim.outputHeight,
                maxPoolingKernelWidth,
                maxPoolingKernelHeight,
                maxPoolingKernelStrideX,
                maxPoolingKernelStrideY,
                firstConvolutionDim.depth,
                maxPoolingActivationFunction,
                maxPoolingActivationFunctionParameters);

        neuralNet.addFullyConnectedHiddenLayer(
                hiddenNeuronCount,
                hiddenActivationFunction,
                hiddenActivationFunctionParameters);

        neuralNet.randomizeWeights(NeuralNetTestUtility.random);

        NeuralNetTestUtility.runNetworkTrainingTest(
                neuralNet,
                bias,
                trainingSets,
                learningRate,
                maxSteps,
                maxErrorRms,
                noProgressResetThreshold);

    }

    private List<TrainingSet> loadTrainingSet(MnistIdxFile trainingLabels, MnistIdxFile trainingImages) {

        assertEquals(trainingLabels.data.size(), trainingImages.data.size());

        List<TrainingSet> sets = new ArrayList<>();

        for (int i = 0; i < 1000; i++) {

            List<Double> inputs = new ArrayList<>();

            byte[] imageData = trainingImages.data.get(i);

            for (byte pixel : imageData) {

                double value = (double) pixel / (double) Byte.MAX_VALUE;
                inputs.add(value);
            }

            byte[] labelData = trainingLabels.data.get(i);
            byte label = labelData[0];

            List<Double> expectedOutputs = Doubles.asList(new double[10]);
            expectedOutputs.set(label, 1.0);

            TrainingSet set = new TrainingSet(inputs, expectedOutputs);
            sets.add(set);
        }

        return sets;
    }

    private MnistIdxFile loadMnistIdxResourceFile(String path) throws IOException {

        try (InputStream stream = getClass().getResourceAsStream(path)) {
            return loadMnistIdxResourceFile(stream);
        }
    }

    private MnistIdxFile loadMnistIdxResourceFile(InputStream stream) throws IOException {

        // NOTE: this is big endian I guess... doesn't really say here (which would be nice):
        //       https://docs.oracle.com/javase/8/docs/api/java/io/DataInputStream.html, but there is a claim
        //       here that it is: https://stackoverflow.com/questions/13211770/endianness-on-datainputstream
        try (DataInputStream bigEndianStream = new DataInputStream(stream)) {

            // TODO: do this better, format is here: http://yann.lecun.com/exdb/mnist/

            int magicNumber = bigEndianStream.readInt();

            int magic0 = (magicNumber >>> 24) & 0xff;
            int magic1 = (magicNumber >>> 16) & 0xff;

            int dataType = (magicNumber >>> 8) & 0xff;
            int dimensionCount = magicNumber & 0xff;

            if ((magic0 != 0) || (magic1 != 0)) {
                throw new IllegalArgumentException("Invalid MNIST IDX file");
            }

            if (dataType != 0x08) {
                throw new IllegalArgumentException("Only single byte data supported for now...");
            }

            int itemCount;
            int itemWidth;
            int itemHeight;

            switch (dimensionCount) {
                case 1: {
                    itemCount = bigEndianStream.readInt();
                    itemHeight = 1;
                    itemWidth = 1;
                }
                break;

                case 3: {
                    itemCount = bigEndianStream.readInt();
                    itemHeight = bigEndianStream.readInt();
                    itemWidth = bigEndianStream.readInt();
                }
                break;

                default:
                    throw new IllegalArgumentException("Only dimensions counts of 1 and 3 are supported");
            }

            List<byte[]> data = new ArrayList<>();

            int itemSize = itemHeight * itemWidth;

            for (int i = 0; i < itemCount; i++) {

                byte[] curItem = new byte[itemSize];

                if (bigEndianStream.read(curItem) != itemSize) {
                    throw new IllegalArgumentException("Failed to read item");
                }

                data.add(curItem);
            }

            return new MnistIdxFile(itemWidth, itemHeight, data);
        }
    }

    private static class MnistIdxFile {

        public final int itemWidth;
        public final int itemHeight;

        public final List<byte[]> data;

        private MnistIdxFile(int itemWidth, int itemHeight, List<byte[]> data) {
            this.itemWidth = itemWidth;
            this.itemHeight = itemHeight;
            this.data = data;
        }
    }
}
