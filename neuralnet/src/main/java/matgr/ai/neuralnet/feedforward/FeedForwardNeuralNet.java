package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class FeedForwardNeuralNet {

    public final ActivationFunction activationFunction;

    // TODO: this should be immutable...
    public final double[] activationFunctionParameters;

    private final List<FeedForwardNeuronLayer> writableLayers;

    public final List<FeedForwardNeuronLayer> layers;

    public final FeedForwardNeuralNetProperties properties;

    public FeedForwardNeuralNet(int inputCount,
                                int outputCount,
                                int hiddenLayers,
                                int neuronsPerHiddenLayer,
                                ActivationFunction activationFunction,
                                double... activationFunctionParameters) {

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;

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

    private FeedForwardNeuralNet(FeedForwardNeuralNet other) {

        if (other == null) {
            throw new IllegalArgumentException("other neural net not provided");
        }

        activationFunction = other.activationFunction;
        activationFunctionParameters = other.activationFunctionParameters;

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

    public void randomize(RandomGenerator random) {

        for (FeedForwardNeuronLayer layer : layers) {
            layer.randomize(random);
        }
    }

    public List<Double> activate(Collection<Double> inputs, double bias) {

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

                    neuronSum += input * weight;

                    if (Double.isNaN(neuronSum)) {
                        // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                        //       return a status code from this function)... it should be given
                        //       "input" and "weight" (so it can decide what to do based on input values being
                        //       infinite/NaN/etc... if it fails, then set this to 0.0
                        neuronSum = 0.0;
                    }
                }

                neuronSum += (neuron.biasWeight * bias);

                if (Double.isNaN(neuronSum)) {
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                    //       return a status code from this function)... it should be given
                    //       "neuron.biasWeight" and "bias" (so it can decide what to do based on input values being
                    //       infinite/NaN/etc... if it fails, then set this to 0.0
                    neuronSum = 0.0;
                }

                double neuronOutput = activationFunction.compute(neuronSum, activationFunctionParameters);

                if (Double.isNaN(neuronOutput)) {
                    // NOTE: sigmoid shouldn't produce NaN, so fallback to this one for now...
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and return a
                    //       status code from this function)... if it fails, then try this
                    neuronOutput = KnownActivationFunctions.SIGMOID.compute(
                            neuronSum,
                            KnownActivationFunctions.SIGMOID.defaultParameters());
                }

                currentOutputs.add(neuronOutput);
            }

            currentInputs = currentOutputs;
            currentOutputs = new ArrayList<>();
        }

        return currentInputs;
    }

    public void backPropagate(double learningRate, List<Double> outputs, List<Double> expectedOutputs) {

        //double error = computeError(outputs, expectedOutputs);

        // TODO: handle NaNs

        List<Double> curOutputs = outputs;
        List<Double> curExpectedOutputs = expectedOutputs;

        for (int layerIndex = writableLayers.size() - 1; layerIndex >= 0; layerIndex--) {

            FeedForwardNeuronLayer layer = writableLayers.get(layerIndex);

            List<Double> nextOutputs = new ArrayList<>();
            List<Double> nextExpectedOutputs = new ArrayList<>();

            for (int neuronIndex = 0; neuronIndex < layer.neurons.size(); neuronIndex++) {

                FeedForwardNeuron neuron = layer.neurons.get(neuronIndex);

                double neuronOutput = curOutputs.get(neuronIndex);
                double neuronExpectedOutput = curExpectedOutputs.get(neuronIndex);

                List<Double> weightDerivatives = new ArrayList<>(layer.weightsCount.total);

                for (int weightIndex = 0; weightIndex < layer.weightsCount.total; weightIndex++) {
                    double currentWegith = layer.
                }

                // TODO: this could maybe be saved during activation? maybe do a NeuronState (like in the acyclic
                //       network)
                double neuronInput = activationFunction.computeInverse(
                        neuronOutput,
                        activationFunctionParameters);

                double expectedNueronInput = activationFunction.computeInverse(
                        neuronExpectedOutput,
                        activationFunctionParameters);

                throw new NotImplementedException();
            }

            curOutputs = nextOutputs;
            curExpectedOutputs = nextExpectedOutputs;
        }
    }

    public double computeError(Collection<Double> outputs, Collection<Double> expectedOutputs) {

        if (properties.outputCount != outputs.size()) {
            throw new IllegalArgumentException("Incorrect number of outputs");
        }
        if (properties.outputCount != expectedOutputs.size()) {
            throw new IllegalArgumentException("Incorrect number of expected outputs");
        }

        Iterator<Double> outputIterator = outputs.iterator();
        Iterator<Double> expectedOutputIterator = expectedOutputs.iterator();

        double error = 0.0;

        while (outputIterator.hasNext()) {

            double output = outputIterator.next();
            double expectedOutput = expectedOutputIterator.next();

            double difference = expectedOutput - output;
            double outputError = 0.5 * (difference * difference);

            error += outputError;
        }

        return error;
    }

    public FeedForwardNeuralNet deepClone() {
        return new FeedForwardNeuralNet(this);
    }

}