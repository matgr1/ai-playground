package matgr.ai.neuralnet.feedforward;

import matgr.ai.genetic.NumericGenome;
import matgr.ai.neuralnet.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class FeedForwardNeuralNetGenome extends NumericGenome {

    public final ActivationFunction activationFunction;

    public final int inputCount;

    public final int outputCount;

    public final int hiddenLayers;

    public final int neuronsPerHiddenLayer;

    protected FeedForwardNeuralNetGenome(NumericGenome genome,
                                         ActivationFunction activationFunction,
                                         int inputCount,
                                         int outputCount,
                                         int hiddenLayers,
                                         int neuronsPerHiddenLayer) {
        super(genome);

        this.activationFunction = activationFunction;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayers = hiddenLayers;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
    }

    public FeedForwardNeuralNet decode() {

        FeedForwardNeuralNet neuralNet = new FeedForwardNeuralNet(
                activationFunction,
                inputCount,
                outputCount,
                hiddenLayers,
                neuronsPerHiddenLayer);

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
