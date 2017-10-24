package matgr.ai.neuralnet;

import com.google.common.primitives.Doubles;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import matgr.ai.neuralnet.feedforward.ConvolutionDimensions;
import matgr.ai.neuralnet.feedforward.FeedForwardNeuralNet;

import java.util.ArrayList;
import java.util.List;

/**
 * Unit test for simple App.
 */
public class NeuralNetTest extends TestCase {

    /**
     * Create the test case
     *
     * @param testName name of the test case
     */
    public NeuralNetTest(String testName) {

        super(testName);
    }

    /**
     * @return the suite of tests being tested
     */
    public static Test suite() {
        return new TestSuite(NeuralNetTest.class);
    }

    public void testBasicFunction() {

        final int inputCount = 2;
        final int outputCount = 4;

        final int hiddenLayers = 1;
        final int neuronsPerHiddenLayer = 5;

        final int sqrtSetCount = 2;

        final double bias = 1;

        final double learningRate = 0.1;
        final int maxSteps = 1000000;
        final double maxErrorRms = 0.01;

        final int noProgressResetThreshold = 5000;

        final boolean outputApplySoftmax = false;
        ActivationFunction outputActivationFunction = KnownActivationFunctions.TANH;
        double[] outputActivationFunctionParameters = outputActivationFunction.defaultParameters();

        FeedForwardNeuralNet<Neuron> neuralNet = new FeedForwardNeuralNet<>(
                new DefaultNeuronFactory(),
                inputCount,
                outputCount,
                outputApplySoftmax,
                outputActivationFunction,
                outputActivationFunctionParameters);

        ActivationFunction activationFunction = KnownActivationFunctions.TANH;
        double[] activationFunctionParameters = activationFunction.defaultParameters();

        for (int i = 0; i < hiddenLayers; i++) {
            neuralNet.addFullyConnectedHiddenLayer(
                    neuronsPerHiddenLayer,
                    activationFunction,
                    activationFunctionParameters);
        }

        neuralNet.randomizeWeights(NeuralNetTestUtility.random);

        List<TrainingSet> trainingSets = getBasicFunctionTrainingSets(sqrtSetCount);

        NeuralNetTestUtility.runNetworkTrainingTest(
                neuralNet,
                bias,
                trainingSets,
                learningRate,
                maxSteps,
                maxErrorRms,
                noProgressResetThreshold);
    }

    private List<TrainingSet> getBasicFunctionTrainingSets(int sqrtCount) {

        List<TrainingSet> sets = new ArrayList<>();

        for (int i = 0; i < sqrtCount; i++) {

            for (int j = 0; j < sqrtCount; j++) {

                List<Double> inputs = new ArrayList<>();
                List<Double> expectedOutputs = new ArrayList<>();

                inputs.add((double) i);
                inputs.add((double) j);

                double xorOutput = (double) (i ^ j);
                // TODO: can only produce numbers in the range 0-1, so &-ing with 0x1 for now...
                double nandOutput = (double) (0x1 & ~(i & j));
                double andOutput = (double) (i & j);
                double orOutput = (double) (i | j);

                expectedOutputs.add(xorOutput);
                expectedOutputs.add(nandOutput);
                expectedOutputs.add(andOutput);
                expectedOutputs.add(orOutput);

                TrainingSet set = new TrainingSet(inputs, expectedOutputs);
                sets.add(set);
            }
        }

        return sets;
    }

    public void testConvolutional() {

        final int numSets = 20;

        final int convolutionInputWidth = 25;
        final int convolutionInputHeight = 25;
        final int convolutionInputDepth = 1;

        final int convolutionalKernelWidth = 6;
        final int convolutionalKernelHeight = 6;
        final int convolutionalKernelDepth = 1;

        final int convolutionInstances = 1;

        final int maxPoolingKernelWidth = 2;
        final int maxPoolingKernelHeight = 2;
        final int maxPoolingKernelDepth = 1;

        final int maxPoolingKernelStrideX = 2;
        final int maxPoolingKernelStrideY = 2;
        final int maxPoolingKernelStrideZ = 1;

        final int outputCount = numSets;

        final double bias = 1;

        final double learningRate = 0.1;
        final int maxSteps = 1000000;
        final double maxErrorRms = 0.1;

        final int noProgressResetThreshold = 5000;

        final ConvolutionDimensions convolutionDimensions = new ConvolutionDimensions(
                convolutionInputWidth,
                convolutionInputHeight,
                convolutionInputDepth,
                convolutionalKernelWidth,
                convolutionalKernelHeight,
                convolutionalKernelDepth);

        final int inputCount = convolutionDimensions.inputWidth * convolutionDimensions.inputHeight;

        ActivationFunction convolutionalActivationFunction = KnownActivationFunctions.IDENTITY;
        double[] convolutionalActivationFunctionParameters = convolutionalActivationFunction.defaultParameters();

        // TODO: one of these should be RELU?
        ActivationFunction maxPoolingActivationFunction = KnownActivationFunctions.RELU;
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

        neuralNet.addConvolutionalHiddenLayer(
                convolutionInputWidth,
                convolutionInputHeight,
                convolutionInputDepth,
                convolutionalKernelWidth,
                convolutionalKernelHeight,
                convolutionalKernelDepth,
                convolutionInstances,
                convolutionalActivationFunction,
                convolutionalActivationFunctionParameters);

        neuralNet.addMaxPoolingHiddenLayer(
                convolutionDimensions.outputWidth,
                convolutionDimensions.outputHeight,
                convolutionDimensions.outputDepth,
                maxPoolingKernelWidth,
                maxPoolingKernelHeight,
                maxPoolingKernelDepth,
                maxPoolingKernelStrideX,
                maxPoolingKernelStrideY,
                maxPoolingKernelStrideZ,
                convolutionInstances,
                maxPoolingActivationFunction,
                maxPoolingActivationFunctionParameters);

        neuralNet.addFullyConnectedHiddenLayer(
                hiddenNeuronCount,
                hiddenActivationFunction,
                hiddenActivationFunctionParameters);

        neuralNet.randomizeWeights(NeuralNetTestUtility.random);

        List<TrainingSet> trainingSets = getConvolutionalTrainingSets(convolutionDimensions, numSets);

        NeuralNetTestUtility.runNetworkTrainingTest(
                neuralNet,
                bias,
                trainingSets,
                learningRate,
                maxSteps,
                maxErrorRms,
                noProgressResetThreshold);
    }

    private List<TrainingSet> getConvolutionalTrainingSets(ConvolutionDimensions convolutionDimensions,
                                                           int numSets) {

        List<TrainingSet> sets = new ArrayList<>();

        int inputWidth = convolutionDimensions.inputWidth;
        int inputHeight = convolutionDimensions.inputHeight;

        int maxIndex = ((inputHeight - 1) * inputWidth) + (inputWidth - 1);

        for (int i = 0; i < numSets; i++) {

            // TODO: real data...

            int divisions = (i + 1);

            double[] inputData = new double[inputHeight * inputWidth];

            for (int y = 0; y < inputHeight; y++) {

                int rowIndex = y * inputWidth;

                for (int x = 0; x < inputWidth; x++) {

                    int index = rowIndex + x;

                    double fractionalValue = (double) index / (double) (maxIndex);
                    double value = Math.round(fractionalValue * (double) divisions) / (double) divisions;

                    inputData[index] = value;
                }
            }

            List<Double> inputs = Doubles.asList(inputData);


            double[] expectedOutputsArray = new double[numSets];
            List<Double> expectedOutputs = Doubles.asList(expectedOutputsArray);

            expectedOutputsArray[i] = 1.0;


//            List<Double> expectedOutputs = new ArrayList<>();
//
//            double output1 = divisions / (double) numSets;
//            double output2 = 1.0 - output1;
//
//            expectedOutputs.add(output1);
//            expectedOutputs.add(output2);


            TrainingSet set = new TrainingSet(inputs, expectedOutputs);
            sets.add(set);
        }

        return sets;
    }
}
