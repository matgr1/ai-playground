package matgr.ai.neuralnet.feedforward;

import matgr.ai.math.RandomFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class FeedForwardNeuron {

    private final List<Double> writableIncomingWeights;

    public final List<Double> incomingWeights;

    public double incomingBiasWeight;

    double preSynapse;
    double postSynapse;

    public FeedForwardNeuron(int inputCount) {

        this.writableIncomingWeights = Arrays.asList(new Double[inputCount]);
        this.incomingWeights = Collections.unmodifiableList(writableIncomingWeights);

        this.preSynapse = 0.0;
        this.postSynapse = 0.0;
    }

    private FeedForwardNeuron(FeedForwardNeuron other) {

        if (null == other) {
            throw new IllegalArgumentException("other neuron not provided");
        }

        this.writableIncomingWeights = new ArrayList<>();
        this.incomingWeights = Collections.unmodifiableList(writableIncomingWeights);

        this.writableIncomingWeights.addAll(other.incomingWeights);

        this.incomingBiasWeight = other.incomingBiasWeight;

        this.preSynapse = other.preSynapse;
        this.postSynapse = other.postSynapse;
    }

    public void randomize(RandomGenerator random) {

        for (int i = 0; i < this.incomingWeights.size(); i++) {
            this.writableIncomingWeights.set(i, getInitialWeight(random));
        }

        this.incomingBiasWeight = getInitialWeight(random);
    }

    public FeedForwardNeuron deepClone() {
        return new FeedForwardNeuron(this);
    }

    public void setIncomingWeight(int index, double weight) {
        writableIncomingWeights.set(index, weight);
    }

    public void setIncomingWeights(List<Double> weights, double biasWeight) {

        if (weights.size() != this.writableIncomingWeights.size()) {
            throw new IllegalArgumentException("Incorrect number of writableIncomingWeights");
        }

        for (int i = 0; i < weights.size(); i++) {
            this.writableIncomingWeights.set(i, weights.get(i));
        }

        this.incomingBiasWeight = biasWeight;

    }

    private static double getInitialWeight(RandomGenerator random) {
        return RandomFunctions.nextDouble(random, -1.0, 1.0);
    }
}
