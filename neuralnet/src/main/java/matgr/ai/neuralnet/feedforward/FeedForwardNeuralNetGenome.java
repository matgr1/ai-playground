package matgr.ai.neuralnet.feedforward;

import matgr.ai.genetic.NumericGenome;
import matgr.ai.neuralnet.activation.ActivationFunction;

import java.util.ArrayList;
import java.util.List;

public abstract class FeedForwardNeuralNetGenome extends NumericGenome {

    public final ActivationFunction activationFunction;

    // TODO: this should be immutable
    public final double[] activationFunctionParameters;

    public final int inputCount;

    public final int outputCount;

    public final int hiddenLayers;

    public final int neuronsPerHiddenLayer;

    protected FeedForwardNeuralNetGenome(NumericGenome genome,
                                         int inputCount,
                                         int outputCount,
                                         int hiddenLayers,
                                         int neuronsPerHiddenLayer,
                                         ActivationFunction activationFunction,
                                         double... activationFunctionParameters) {
        super(genome);

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;
        this.inputCount = inputCount;
        this.outputCount = outputCount;
        this.hiddenLayers = hiddenLayers;
        this.neuronsPerHiddenLayer = neuronsPerHiddenLayer;
    }

    public FeedForwardNeuralNet decode() {

        FeedForwardNeuralNet neuralNet = new FeedForwardNeuralNet(
                inputCount,
                outputCount,
                hiddenLayers,
                neuronsPerHiddenLayer,
                activationFunction,
                activationFunctionParameters);

        int weightIndex = 0;

        for (FeedForwardNeuronLayer layer : neuralNet.layers) {

            for (FeedForwardNeuron neuron : layer.neurons) {

                List<Double> weights = new ArrayList<>();

                for (int i = 0; i < neuron.incomingWeights.size(); i++) {
                    weights.add(this.getGene(weightIndex++));
                }

                double bias = this.getGene(weightIndex++);

                neuron.setIncomingWeights(weights, bias);
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
