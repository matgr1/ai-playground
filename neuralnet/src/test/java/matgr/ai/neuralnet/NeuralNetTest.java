package matgr.ai.neuralnet;

import junit.framework.Test;
import junit.framework.TestCase;
import junit.framework.TestSuite;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import matgr.ai.neuralnet.feedforward.FeedForwardNeuralNet;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

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

    // TODO: turn these into actual tests...
    public void test1() {

        final int inputCount = 2;
        final int outputCount = 4;

        final int hiddenLayers = 1;
        final int neuronsPerHiddenLayer = 5;

        final int maxSteps = 1000000;
        final double maxError = 0.0001;

        final int sqrtSetCount = 2;

        final double bias = 1;

        final double learningRate = 0.5;

        ActivationFunction activationFunction = KnownActivationFunctions.SIGMOID;
        double[] activationFunctionParameters = activationFunction.defaultParameters();

        List<NeuronParameters> outputParameters = new ArrayList<>();
        for (int i = 0; i < outputCount; i++) {
            outputParameters.add(new NeuronParameters(activationFunction, activationFunctionParameters));
        }

        RandomGenerator random = new MersenneTwister();

        FeedForwardNeuralNet<Neuron> neuralNet = new FeedForwardNeuralNet<Neuron>(
                new DefaultNeuronFactory(),
                inputCount,
                outputParameters);

        List<NeuronParameters> hiddenLayerParameters = new ArrayList<>();
        for (int i = 0; i < neuronsPerHiddenLayer; i++) {
            hiddenLayerParameters.add(new NeuronParameters(activationFunction, activationFunctionParameters));
        }

        for (int i = 0; i < hiddenLayers; i++) {
            neuralNet.addHiddenLayer(hiddenLayerParameters);
        }

        neuralNet.randomizeWeights(random);

        List<TrainingSet> trainingSets = getTrainingSets(sqrtSetCount);

        double error = Double.POSITIVE_INFINITY;
        double bestError = Double.POSITIVE_INFINITY;

        for (int step = 0; step < maxSteps; step++) {

            double curStepMaxError = Double.NEGATIVE_INFINITY;

            for (TrainingSet set : trainingSets) {

                neuralNet.activate(set.inputs, bias);

                error = neuralNet.getCurrentError(set.expectedOutputs);
                curStepMaxError = Math.max(curStepMaxError, error);

                // TODO: put this back?
                //if (error > maxError) {
                neuralNet.backPropagate(learningRate, bias, set.expectedOutputs);
                //}
            }

            error = curStepMaxError;

            bestError = Math.min(bestError, curStepMaxError);

            // TODO: error calculation should maybe be different... maybe max for any individual output/set?
            if (error < maxError) {
                break;
            }
        }

        // print results...
        for (TrainingSet set : trainingSets) {

            neuralNet.activate(set.inputs, bias);

            List<Double> outputs = neuralNet.getCurrentOutputs();
            double err = neuralNet.getCurrentError(set.expectedOutputs);

            System.out.println(set.inputs);
            System.out.println(set.expectedOutputs);
            System.out.println(outputs);
            System.out.println(err);
            System.out.println();
        }

        assertTrue(error < maxError);
    }

    private List<TrainingSet> getTrainingSets(int sqrtCount) {

        List<TrainingSet> sets = new ArrayList<>();

        for (int i = 0; i < sqrtCount; i++) {

            for (int j = 0; j < sqrtCount; j++) {

                TrainingSet set = new TrainingSet(i, j);
                sets.add(set);

            }
        }

        return sets;
    }

    private static class TrainingSet {
        public List<Double> inputs;
        public List<Double> expectedOutputs;

        public TrainingSet(int inputA, int inputB) {
            inputs = new ArrayList<>();
            expectedOutputs = new ArrayList<>();

            inputs.add((double) inputA);
            inputs.add((double) inputB);

            double xorOutput = (double) (inputA ^ inputB);
            // TODO: can only produce numbers in the range 0-1, so &-ing with 0x1 for now...
            double nandOutput = (double) (0x1 & ~(inputA & inputB));
            double andOutput = (double) (inputA & inputB);
            double orOutput = (double) (inputA | inputB);

            expectedOutputs.add(xorOutput);
            expectedOutputs.add(nandOutput);
            expectedOutputs.add(andOutput);
            expectedOutputs.add(orOutput);
        }
    }
}
