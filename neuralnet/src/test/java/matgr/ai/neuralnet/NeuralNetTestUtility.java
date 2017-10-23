package matgr.ai.neuralnet;

import junit.framework.Assert;
import matgr.ai.math.MathFunctions;
import matgr.ai.neuralnet.feedforward.ErrorType;
import matgr.ai.neuralnet.feedforward.FeedForwardNeuralNet;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetTestUtility extends Assert {

    public static final RandomGenerator random = new MersenneTwister();

    public static void runNetworkTrainingTest(FeedForwardNeuralNet<Neuron> neuralNet,
                                              double bias,
                                              List<TrainingSet> trainingSets,
                                              double learningRate,
                                              int maxSteps,
                                              double maxErrorRms,
                                              int noProgressResetThreshold) {

        double bestErrorRms = Double.POSITIVE_INFINITY;
        double lastErrorRms = Double.POSITIVE_INFINITY;

        int noProgressCount = 0;

        long startNs = System.nanoTime();
        long prevNs = startNs;

        long printPeriodNs = 1000000000;

        int step = 0;

        for (; step < maxSteps; step++) {

            double errorRms = Double.NEGATIVE_INFINITY;

            // TODO: this is stochastic gradient descent... should probably do mini-batches... for mini-batches, it
            //       might be as simple as keeping track of the sum of the dE/DW's and applying the weight adjustment
            //       at the end (dividing by batch size) (see here: http://neuralnetworksanddeeplearning.com/chap2.html)
            // TODO: adaptive learning rate
            List<TrainingSet> trainingSetsList = new ArrayList<>(trainingSets);

            while (trainingSetsList.size() > 0) {

                int index = random.nextInt(trainingSetsList.size());

                TrainingSet set = trainingSetsList.remove(index);

                neuralNet.activate(set.inputs, bias);

                double setErrorRms = neuralNet.getCurrentError(set.expectedOutputs, ErrorType.Rms);
                errorRms = Math.max(errorRms, setErrorRms);

                if (errorRms > maxErrorRms) {
                    neuralNet.backPropagate(learningRate, bias, set.expectedOutputs);
                }
            }

            if (MathFunctions.fuzzyCompare(lastErrorRms, errorRms) || (errorRms > bestErrorRms)) {

                noProgressCount++;

            } else {

                noProgressCount = 0;
            }

            lastErrorRms = errorRms;
            bestErrorRms = Math.min(bestErrorRms, errorRms);

            long nowNs = System.nanoTime();
            if ((nowNs - prevNs) > printPeriodNs) {

                System.out.println(String.format("Executed %d steps - current error (RMS): %.6f", step + 1, errorRms));
                prevNs = nowNs;
            }

            // TODO: error calculation should maybe be different... maybe max for any individual output/set?
            if (errorRms < maxErrorRms) {
                break;
            }

            if (noProgressCount >= noProgressResetThreshold) {

                System.out.println(String.format("No forward progress after %d steps, resetting...", noProgressCount));

                neuralNet.randomizeWeights(random);
                noProgressCount = 0;
            }
        }

        // print results...

        long nsTaken = System.nanoTime() - startNs;
        double sTaken = (double) nsTaken / 1000000000.0;
        double stepsPerS = step / sTaken;

        System.out.println();
        System.out.println(
                String.format(
                        "Executed %d steps in %.6f seconds (%.2f steps per second) - final error (RMS): %.6f",
                        step,
                        sTaken,
                        stepsPerS,
                        lastErrorRms));

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

        assertTrue(lastErrorRms < maxErrorRms);
    }
}
