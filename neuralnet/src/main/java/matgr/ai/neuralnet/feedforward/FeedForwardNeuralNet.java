package matgr.ai.neuralnet.feedforward;

import com.google.common.primitives.Doubles;
import matgr.ai.neuralnet.activation.ActivationFunction;
import matgr.ai.neuralnet.activation.KnownActivationFunctions;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.*;

// TODO: can maybe share more with cyclic?
public class FeedForwardNeuralNet {

    private final List<FeedForwardNeuron> writableInputNeurons;
    private final List<FeedForwardNeuronLayer> writableLayers;

    public final ActivationFunction activationFunction;

    // TODO: this should be immutable...
    public final double[] activationFunctionParameters;

    public final List<FeedForwardNeuron> inputNeurons;

    public final List<FeedForwardNeuronLayer> layers;

    public final FeedForwardNeuralNetProperties properties;

    // TODO: allow different numbers of neurons per hidden layer
    public FeedForwardNeuralNet(int inputCount,
                                int outputCount,
                                int hiddenLayers,
                                int neuronsPerHiddenLayer,
                                ActivationFunction activationFunction,
                                double... activationFunctionParameters) {

        if (activationFunction == null) {
            throw new IllegalArgumentException("Activation function net not provided");
        }
        if (activationFunctionParameters == null) {
            throw new IllegalArgumentException("Activation function parameters net not provided");
        }

        this.activationFunction = activationFunction;
        this.activationFunctionParameters = activationFunctionParameters;

        this.writableInputNeurons = new ArrayList<>();
        this.inputNeurons = Collections.unmodifiableList(this.writableInputNeurons);

        this.writableLayers = new ArrayList<>();
        this.layers = Collections.unmodifiableList(this.writableLayers);

        int hiddenNeurons = 0;
        WeightsCount hiddenNeuronWeights = new WeightsCount();

        int outputNeurons = 0;
        WeightsCount outputNeuronWeights = new WeightsCount();

        for (int i = 0; i < inputCount; i++) {

            FeedForwardNeuron inputNeuron = new FeedForwardNeuron(0);
            writableInputNeurons.add(inputNeuron);
        }

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
            throw new IllegalArgumentException("Other neural net not provided");
        }

        this.activationFunction = other.activationFunction;
        this.activationFunctionParameters = other.activationFunctionParameters;

        this.writableInputNeurons = new ArrayList<>();
        this.inputNeurons = Collections.unmodifiableList(this.writableInputNeurons);

        this.writableLayers = new ArrayList<>();
        this.layers = Collections.unmodifiableList(this.writableLayers);

        for (FeedForwardNeuron neuron : other.writableInputNeurons) {
            writableInputNeurons.add(neuron.deepClone());
        }

        for (FeedForwardNeuronLayer layer : other.writableLayers) {
            writableLayers.add(layer.deepClone());
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

        // initialize inputs
        Iterator<Double> inputIterator = inputs.iterator();
        for (FeedForwardNeuron inputNeuron : writableInputNeurons) {

            inputNeuron.postSynapse = inputIterator.next();
        }

        List<FeedForwardNeuron> previousNeurons = writableInputNeurons;

        for (FeedForwardNeuronLayer layer : writableLayers) {

            // TODO: this could maybe be done in one of the other loops?
            for (FeedForwardNeuron neuron : layer.neurons) {
                neuron.preSynapse = 0.0;
                neuron.postSynapse = 0.0;
            }

            for (FeedForwardNeuron neuron : layer.neurons) {

                for (int i = 0; i < previousNeurons.size(); i++) {

                    FeedForwardNeuron previousNeuron = previousNeurons.get(i);
                    double weight = neuron.incomingWeights.get(i);

                    neuron.preSynapse += previousNeuron.postSynapse * weight;

                    if (Double.isNaN(neuron.preSynapse)) {
                        // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                        //       return a status code from this function)... it should be given
                        //       "input" and "weight" (so it can decide what to do based on input values being
                        //       infinite/NaN/etc... if it fails, then set this to 0.0
                        neuron.preSynapse = 0.0;
                    }
                }

                neuron.preSynapse += (neuron.incomingBiasWeight * bias);

                if (Double.isNaN(neuron.preSynapse)) {
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and
                    //       return a status code from this function)... it should be given
                    //       "neuron.incomingBiasWeight" and "bias" (so it can decide what to do based on input values being
                    //       infinite/NaN/etc... if it fails, then set this to 0.0
                    neuron.preSynapse = 0.0;
                }

                neuron.postSynapse = activationFunction.compute(neuron.preSynapse, activationFunctionParameters);

                if (Double.isNaN(neuron.postSynapse)) {
                    // NOTE: sigmoid shouldn't produce NaN, so fallback to this one for now...
                    // TODO: pass in some sort of NaN handler (with the ability to completely bail out and return a
                    //       status code from this function)... if it fails, then try this
                    neuron.postSynapse = KnownActivationFunctions.SIGMOID.compute(
                            neuron.preSynapse,
                            KnownActivationFunctions.SIGMOID.defaultParameters());
                }
            }

            previousNeurons = layer.neurons;
        }

        // copy the outputs (note that "previousNeurons" should now be the outputs)
        List<Double> outputs = new ArrayList<>();

        for (FeedForwardNeuron neuron : previousNeurons) {
            outputs.add(neuron.postSynapse);
        }

        return outputs;
    }

    public void backPropagate(double learningRate, double bias, List<Double> expectedOutputs) {

        // TODO: handle NaNs

        int outputLayerIndex = writableLayers.size() - 1;

        FeedForwardNeuronLayer layer = writableLayers.get(outputLayerIndex);
        double[] dE_dOut_previous = null;

        for (int layerIndex = outputLayerIndex; layerIndex >= 0; layerIndex--) {

            FeedForwardNeuronLayer previousLayer = null;

            List<FeedForwardNeuron> previousNeurons;
            if (layerIndex > 0) {

                previousLayer = writableLayers.get(layerIndex - 1);
                previousNeurons = previousLayer.neurons;

            } else {

                previousNeurons = inputNeurons;
            }

            double[] dE_dOut_hidden = new double[previousNeurons.size()];

            for (int neuronIdx = 0; neuronIdx < layer.neurons.size(); neuronIdx++) {

                FeedForwardNeuron neuron = layer.neurons.get(neuronIdx);

                double neuronOutput = neuron.postSynapse;

                double dE_dOut;

                // TODO: try to either move this test out a loop or do this better...
                if (layerIndex == outputLayerIndex) {

                    double neuronExpectedOutput = expectedOutputs.get(neuronIdx);
                    dE_dOut = -(neuronExpectedOutput - neuronOutput);

                }else{

                    dE_dOut = dE_dOut_previous[neuronIdx];
                }

                double dOut_dIn = activationFunction.computeDerivativeFromActivationOutput(
                        neuronOutput,
                        activationFunctionParameters);

                double dE_dIn = dE_dOut * dOut_dIn;

                // update incoming connection weights
                for (int previousNeuronIdx = 0; previousNeuronIdx < previousNeurons.size(); previousNeuronIdx++) {

                    FeedForwardNeuron previousNeuron = previousNeurons.get(previousNeuronIdx);

                    double dIn_dW = previousNeuron.postSynapse;
                    double dE_dW = dE_dIn * dIn_dW;

                    double currentWeight = neuron.incomingWeights.get(previousNeuronIdx);

                    // TODO: don't need to compute this on the last pass
                    dE_dOut_hidden[previousNeuronIdx] += (dE_dIn * currentWeight);

                    double newWeight = currentWeight - (dE_dW * learningRate);
                    neuron.setIncomingWeight(previousNeuronIdx, newWeight);
                }

                // update incoming bias weight
                double dIn_dW_Bias = bias;
                double dE_dW_Bias = dE_dIn * dIn_dW_Bias;

                double currentWeight_Bias = neuron.incomingBiasWeight;
                double newWeight_Bias = currentWeight_Bias - (dE_dW_Bias * learningRate);

                neuron.incomingBiasWeight = newWeight_Bias;
            }

            layer = previousLayer;
            dE_dOut_previous = dE_dOut_hidden;
        }
    }

    public List<Double> getCurrentOutputs() {

        int outputLayerIndex = writableLayers.size() - 1;
        FeedForwardNeuronLayer outputLayer = writableLayers.get(outputLayerIndex);

        List<Double> outputs = new ArrayList<>();

        for (FeedForwardNeuron outputNeuron : outputLayer.neurons) {

            double output = outputNeuron.postSynapse;
            outputs.add(output);
        }

        return outputs;
    }

    public double getCurrentError(Collection<Double> expectedOutputs) {

        if (properties.outputCount != expectedOutputs.size()) {
            throw new IllegalArgumentException("Incorrect number of expected outputs");
        }

        int outputLayerIndex = writableLayers.size() - 1;

        FeedForwardNeuronLayer outputLayer = writableLayers.get(outputLayerIndex);

        Iterator<Double> expectedOutputIterator = expectedOutputs.iterator();

        // TODO: which to use for this?
        //double sumOfErrorSquares = 0.0;
        double error = 0.0;

        for (FeedForwardNeuron outputNeuron : outputLayer.neurons) {

            double output = outputNeuron.postSynapse;
            double expectedOutput = expectedOutputIterator.next();

            double difference = expectedOutput - output;
            double outputError = (difference * difference);

            error += (0.5 * outputError);
            //sumOfErrorSquares += outputError;
        }

        //return Math.sqrt(sumOfErrorSquares);
        return error;
    }

    public FeedForwardNeuralNet deepClone() {
        return new FeedForwardNeuralNet(this);
    }

}