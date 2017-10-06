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
        final int outputCount = 2;

        final int hiddenLayers = 1;
        final int neuronsPerHiddenLayer = 4;

        final int maxSteps = 100;
        final double maxError = 0.01;

        final int sqrtSetCount = 2;

        final double bias = 1;

        final double learningRate = 0.01;

        ActivationFunction activationFunction = KnownActivationFunctions.SIGMOID;
        double[] activationFunctionParameters = activationFunction.defaultParameters();

        RandomGenerator random = new MersenneTwister();

        FeedForwardNeuralNet neuralNet = new FeedForwardNeuralNet(
                inputCount,
                outputCount,
                hiddenLayers,
                neuronsPerHiddenLayer,
                activationFunction,
                activationFunctionParameters);

        neuralNet.randomize(random);

        List<TrainingSet> trainingSets = getTrainingSets(sqrtSetCount);

        for (int step = 0; step < maxSteps; step++) {

            for (TrainingSet set : trainingSets) {

                double error = Double.POSITIVE_INFINITY;

                List<Double> outputs = neuralNet.activate(set.inputs, bias);

                error = neuralNet.computeError(outputs, set.expectedOutputs);

                if (error < maxError) {
                    break;
                }

                neuralNet.backPropagate(learningRate, outputs, set.expectedOutputs);
            }
        }


        //assertTrue(error < maxError);
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
            double andOutput = (double) (inputA & inputB);

            expectedOutputs.add(xorOutput);
            expectedOutputs.add(andOutput);
        }
    }
}
