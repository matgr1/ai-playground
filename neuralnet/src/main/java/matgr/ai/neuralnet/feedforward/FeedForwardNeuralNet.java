package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.activation.SoftplusActivationFunction;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class FeedForwardNeuralNet {

    // TODO: allow different activation functions ...also, for NEAT the default activationResponse
    //       should be 4.9 (see here: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
    private final double activationResponse;

    private final List<FeedForwardNeuronLayer> writableLayers;

    public final List<FeedForwardNeuronLayer> layers;

    public final FeedForwardNeuralNetProperties properties;

    public FeedForwardNeuralNet(int inputCount,
                                int outputCount,
                                int hiddenLayers,
                                int neuronsPerHiddenLayer,
                                double activationResponse) {

        this.activationResponse = activationResponse;

        writableLayers = new ArrayList<>();
        layers = Collections.unmodifiableList(writableLayers);

        int hiddenNeurons = 0;
        WeightsCount hiddenNeuronWeights = new WeightsCount();

        int outputNeurons = 0;
        WeightsCount outputNeuronWeights = new WeightsCount();

        if (hiddenLayers > 0) {

            FeedForwardNeuronLayer firstHiddenLayer = new FeedForwardNeuronLayer(
                    neuronsPerHiddenLayer,
                    inputCount);
            writableLayers.add(firstHiddenLayer);

            hiddenNeurons += firstHiddenLayer.neurons.size();
            hiddenNeuronWeights = hiddenNeuronWeights.add(firstHiddenLayer.weightsCount);

            for (int i = 0; i < hiddenLayers - 1; ++i) {

                FeedForwardNeuronLayer hiddenLayer = new FeedForwardNeuronLayer(
                        neuronsPerHiddenLayer,
                        neuronsPerHiddenLayer);
                writableLayers.add(hiddenLayer);

                hiddenNeurons += hiddenLayer.neurons.size();
                hiddenNeuronWeights = hiddenNeuronWeights.add(hiddenLayer.weightsCount);

            }

            FeedForwardNeuronLayer outputLayer = new FeedForwardNeuronLayer(
                    outputCount,
                    neuronsPerHiddenLayer);
            writableLayers.add(outputLayer);

            outputNeurons += outputLayer.neurons.size();
            outputNeuronWeights = outputNeuronWeights.add(outputLayer.weightsCount);

        } else {

            FeedForwardNeuronLayer outputLayer = new FeedForwardNeuronLayer(
                    outputCount,
                    inputCount);
            writableLayers.add(outputLayer);

            outputNeurons += outputLayer.neurons.size();
            outputNeuronWeights = outputNeuronWeights.add(outputLayer.weightsCount);

        }

        properties = new FeedForwardNeuralNetProperties(
                inputCount,
                outputCount,
                hiddenLayers,
                neuronsPerHiddenLayer,
                hiddenNeurons,
                hiddenNeuronWeights,
                outputNeurons,
                outputNeuronWeights);
    }

    public void randomize(RandomGenerator random) {

        for (FeedForwardNeuronLayer layer : layers) {
            layer.randomize(random);
        }

    }

    private FeedForwardNeuralNet(FeedForwardNeuralNet other) {

        if (other == null) {
            throw new IllegalArgumentException("other neural net not provided");
        }

        this.activationResponse = other.activationResponse;

        writableLayers = new ArrayList<>();
        layers = Collections.unmodifiableList(writableLayers);

        for (FeedForwardNeuronLayer layer : other.layers) {
            FeedForwardNeuronLayer copy = layer.deepClone();
            writableLayers.add(copy);
        }

        properties = new FeedForwardNeuralNetProperties(
                other.properties.inputCount,
                other.properties.outputCount,
                other.properties.hiddenLayers,
                other.properties.neuronsPerHiddenLayer,
                other.properties.hiddenNeurons,
                other.properties.hiddenNeuronWeights,
                other.properties.outputNeurons,
                other.properties.outputNeuronWeights);
    }

    public List<Double> activate(List<Double> inputs, double bias) {

        if (properties.inputCount != inputs.size()) {
            throw new IllegalArgumentException("Incorrect number of inputs");
        }

        List<Double> currentInputs = new ArrayList<>();
        currentInputs.addAll(inputs);

        List<Double> currentOutputs = new ArrayList<>();

        for (FeedForwardNeuronLayer layer : writableLayers) {

            for (FeedForwardNeuron neuron : layer.neurons) {

                double neuronSum = 0;

                // NOTE: no need to check that currentInputs.size() == neuron.writableWeights.size() since the
                //       creation of the neural net ensures that
                for (int i = 0; i < currentInputs.size(); i++) {

                    double input = currentInputs.get(i);
                    double weight = neuron.weights.get(i);

                    neuronSum += (input * weight);

                }

                neuronSum += (neuron.biasWeight * bias);

                // TODO: allow different activation functions
                double neuronOutput = SoftplusActivationFunction.instance.compute(neuronSum, activationResponse);
                currentOutputs.add(neuronOutput);
            }

            currentInputs = currentOutputs;
            currentOutputs = new ArrayList<>();
        }

        return currentInputs;
    }

    public FeedForwardNeuralNet deepClone() {
        return new FeedForwardNeuralNet(this);
    }

}