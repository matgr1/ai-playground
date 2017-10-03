package matgr.ai.neuralnet.feedforward;

import matgr.ai.genetic.NumericGenome;

import java.util.ArrayList;
import java.util.List;

public abstract class FeedForwardNeuralNetGenome extends NumericGenome {

    public final int inputCount;

    public final int outputCount;

    public final int hiddenLayers;

    public final int neuronsPerHiddenLayer;

    public final double activationResponse;

    protected FeedForwardNeuralNetGenome(NumericGenome genome,
                                         int inputCount,
                                         int outputCount,
                                         int hiddenLayers,
                                         int neuronsPerHiddenLayer,
                                         double activationResponse) {
        super(genome);

        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayers = hiddenLayers;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
        this.activationResponse = activationResponse;
    }

    public FeedForwardNeuralNet decode() {

        FeedForwardNeuralNet neuralNet = new FeedForwardNeuralNet(
                inputCount,
                outputCount,
                hiddenLayers,
                neuronsPerHiddenLayer,
                activationResponse);

        int weightIndex = 0;

        for (FeedForwardNeuronLayer layer : neuralNet.layers) {

            for (FeedForwardNeuron neuron : layer.neurons) {

                List<Double> weights = new ArrayList<>();

                for (int i = 0; i < neuron.weights.size(); i++) {
                    weights.add(this.getGene(weightIndex++));
                }

                double bias = this.getGene(weightIndex++);

                neuron.setWeights(weights, bias);

            }

        }

        // TODO: would be nice to check this in constructor as well... even better to make it so the numbers
        //       can't get out of sync...
        if (weightIndex != this.geneCount()) {
            throw new IllegalStateException("Incorrect genome/neural net size");
        }

        return neuralNet;
    }
}
