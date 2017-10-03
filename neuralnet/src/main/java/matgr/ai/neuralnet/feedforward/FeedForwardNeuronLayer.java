package matgr.ai.neuralnet.feedforward;

import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FeedForwardNeuronLayer {

    public final List<FeedForwardNeuron> neurons;

    public final WeightsCount weightsCount;

    public FeedForwardNeuronLayer(int neuronCount, int inputsPerNeuron) {

        List<FeedForwardNeuron> writableNeurons = new ArrayList<>();
        neurons = Collections.unmodifiableList(writableNeurons);

        int neuronWeights = 0;
        int biasWeights = 0;

        for (int i = 0; i < neuronCount; i++) {
            FeedForwardNeuron neuron = new FeedForwardNeuron(inputsPerNeuron);
            writableNeurons.add(neuron);

            neuronWeights += neuron.weights.size();
            biasWeights++;
        }

        weightsCount = new WeightsCount(neuronWeights, biasWeights);
    }

    private FeedForwardNeuronLayer(FeedForwardNeuronLayer other) {
        if (null == other) {
            throw new IllegalArgumentException("other");
        }

        List<FeedForwardNeuron> writableNeurons = new ArrayList<>();
        neurons = Collections.unmodifiableList(writableNeurons);

        for (FeedForwardNeuron neuron : other.neurons) {
            FeedForwardNeuron copy = neuron.deepClone();
            writableNeurons.add(copy);
        }

        weightsCount = other.weightsCount;
    }

    public void randomize(RandomGenerator random) {

        for (FeedForwardNeuron neuron : neurons) {
            neuron.randomize(random);
        }
    }

    public FeedForwardNeuronLayer deepClone() {
        return new FeedForwardNeuronLayer(this);
    }

}
