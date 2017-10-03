package matgr.ai.neuralnet.feedforward;

import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class FeedForwardNeuron {

    private final List<Double> writableWeights;

    public final List<Double> weights;

    public double biasWeight;

    public FeedForwardNeuron(int inputCount) {

        this.writableWeights = Arrays.asList(new Double[inputCount]);
        this.weights = Collections.unmodifiableList(writableWeights);
    }

    public void randomize(RandomGenerator random){

        for (int i = 0; i < this.weights.size(); i++) {
            this.writableWeights.add(getInitialWeight(random));
        }

        this.biasWeight = getInitialWeight(random);
    }

    private FeedForwardNeuron(FeedForwardNeuron other) {

        if (null == other) {
            throw new IllegalArgumentException("other neuron not provided");
        }

        this.writableWeights = new ArrayList<>();
        this.weights = Collections.unmodifiableList(writableWeights);

        this.writableWeights.addAll(other.weights);

        this.biasWeight = other.biasWeight;

    }

    public FeedForwardNeuron deepClone() {
        return new FeedForwardNeuron(this);
    }

    public void setWeights(List<Double> weights, double biasWeight) {

        if (weights.size() != this.writableWeights.size()) {
            throw new IllegalArgumentException("Incorrect number of writableWeights");
        }

        for (int i = 0; i < weights.size(); i++) {
            this.writableWeights.set(i, weights.get(i));
        }

        this.biasWeight = biasWeight;

    }

    private static double getInitialWeight(RandomGenerator random) {
        return RandomFunctions.nextDouble(random, -1.0, 1.0);
    }
}
