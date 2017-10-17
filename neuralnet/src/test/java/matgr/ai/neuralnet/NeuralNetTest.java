package matgr.ai.neuralnet;

import com.google.common.primitives.Doubles;
import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import matgr.ai.neuralnet.feedforward.ConvolutionalLayerSizes;
import matgr.ai.neuralnet.feedforward.ErrorType;
import matgr.ai.neuralnet.feedforward.FeedForwardNeuralNet;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;


/**
 * Unit test for simple App.
 */
public class NeuralNetTest extends TestCase {


    private static final RandomGenerator random = new MersenneTwister();

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

        final double learningRate = 0.5;
        final int maxSteps = 1000000;
        final double maxErrorRms = 0.01;

        final boolean outputApplySoftmax = false;
        ActivationFunction outputActivationFunction = KnownActivationFunctions.SIGMOID;
        double[] outputActivationFunctionParameters = outputActivationFunction.defaultParameters();

        FeedForwardNeuralNet<Neuron> neuralNet = new FeedForwardNeuralNet<>(
                new DefaultNeuronFactory(),
                inputCount,
                outputCount,
                outputApplySoftmax,
                outputActivationFunction,
                outputActivationFunctionParameters);

        ActivationFunction activationFunction = KnownActivationFunctions.SIGMOID;
        double[] activationFunctionParameters = activationFunction.defaultParameters();

        for (int i = 0; i < hiddenLayers; i++) {
            neuralNet.addHiddenLayer(
                    neuronsPerHiddenLayer,
                    activationFunction,
                    activationFunctionParameters);
        }

        neuralNet.randomizeWeights(random);

        List<TrainingSet> trainingSets = getBasicFunctionTrainingSets(sqrtSetCount);
        runNetworkTrainingTest(neuralNet, bias, trainingSets, learningRate, maxSteps, maxErrorRms);
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

        final int convolutionInputWidth = 10;
        final int convolutionInputHeight = 10;

        final int convolutionalKernelRadiusX = 1;
        final int convolutionalKernelRadiusY = 1;

        final int outputCount = 2;

        final double bias = 1;

        final double learningRate = 0.5;
        final int maxSteps = 1000000;
        final double maxErrorRms = 0.01;

        final int numSets = 20;

        final ConvolutionalLayerSizes convolutionalLayerSizes = new ConvolutionalLayerSizes(
                convolutionInputWidth,
                convolutionInputHeight,
                convolutionalKernelRadiusX,
                convolutionalKernelRadiusY);

        final int inputCount = convolutionalLayerSizes.inputWidth * convolutionalLayerSizes.inputHeight;

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

        ActivationFunction convolutionalActivationFunction = KnownActivationFunctions.SIGMOID;
        double[] convolutionalActivationFunctionParameters = convolutionalActivationFunction.defaultParameters();

        neuralNet.addConvolutionalHiddenLayer(
                convolutionInputWidth,
                convolutionInputHeight,
                convolutionalKernelRadiusX,
                convolutionalKernelRadiusY,
                convolutionalActivationFunction,
                convolutionalActivationFunctionParameters);

        neuralNet.randomizeWeights(random);

        List<TrainingSet> trainingSets = getConvolutionalTrainingSets(convolutionalLayerSizes, numSets);
        runNetworkTrainingTest(neuralNet, bias, trainingSets, learningRate, maxSteps, maxErrorRms);
    }

    private List<TrainingSet> getConvolutionalTrainingSets(ConvolutionalLayerSizes convolutionalLayerSizes,
                                                           int numSets) {

        List<TrainingSet> sets = new ArrayList<>();

        int inputWidth = convolutionalLayerSizes.inputWidth;
        int inputHeight = convolutionalLayerSizes.inputHeight;

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

            List<Double> expectedOutputs = new ArrayList<>();

            double output1 = divisions / (double) numSets;
            double output2 = 1.0 - output1;

            expectedOutputs.add(output1);
            expectedOutputs.add(output2);

            TrainingSet set = new TrainingSet(inputs, expectedOutputs);
            sets.add(set);
        }

        return sets;
    }

    private static void runNetworkTrainingTest(FeedForwardNeuralNet<Neuron> neuralNet,
                                               double bias,
                                               List<TrainingSet> trainingSets,
                                               double learningRate,
                                               int maxSteps,
                                               double maxErrorRms) {

        double errorRms = Double.POSITIVE_INFINITY;

        long startNs = System.nanoTime();
        long prevNs = startNs;

        long printPeriodNs = 1000000000;

        int step = 0;
        for (; step < maxSteps; step++) {

            double curStepMaxErrorRms = Double.NEGATIVE_INFINITY;

            for (TrainingSet set : trainingSets) {

                neuralNet.activate(set.inputs, bias);

                errorRms = neuralNet.getCurrentError(set.expectedOutputs, ErrorType.Rms);
                curStepMaxErrorRms = Math.max(curStepMaxErrorRms, errorRms);

                // TODO: put this back?
                //if (error > maxError) {
                neuralNet.backPropagate(learningRate, bias, set.expectedOutputs);
                //}
            }

            errorRms = curStepMaxErrorRms;

            long nowNs = System.nanoTime();
            if ((nowNs - prevNs) > printPeriodNs) {

                System.out.println(String.format("Executed %d steps - current error (RMS): %.6f", step + 1, errorRms));
                prevNs = nowNs;
            }

            // TODO: error calculation should maybe be different... maybe max for any individual output/set?
            if (errorRms < maxErrorRms) {
                break;
            }
        }

        // print results...

        long nsTaken = System.nanoTime() - startNs;
        double sTaken = (double) nsTaken / 1000000000.0;
        double stepsPerS = step / sTaken;

        System.out.println();
        System.out.println(
                String.format(
                        "Executed %d steps in %.6f seconds (%.2f steps per second)",
                        step,
                        sTaken,
                        stepsPerS));

        for (int i = 0; i < trainingSets.size(); i++) {

            TrainingSet set = trainingSets.get(i);

            neuralNet.activate(set.inputs, bias);

            List<Double> outputs = neuralNet.getCurrentOutputs();
            double currErrorRms = neuralNet.getCurrentError(set.expectedOutputs, ErrorType.Rms);

            System.out.println();
            System.out.println(String.format("Training set %d:", i));
            System.out.println(String.format("  inputs            : %s", set.inputs));
            System.out.println(String.format("  expected outputs  : %s", set.expectedOutputs));
            System.out.println(String.format("  outputs           : %s", outputs));
            System.out.println(String.format("  error (RMS)       : %.6f", currErrorRms));
        }

        assertTrue(errorRms < maxErrorRms);
    }

    private static class TrainingSet {

        public List<Double> inputs;
        public List<Double> expectedOutputs;

        public TrainingSet(List<Double> inputs, List<Double> expectedOutputs) {

            this.inputs = inputs;
            this.expectedOutputs = expectedOutputs;
        }
    }
}
