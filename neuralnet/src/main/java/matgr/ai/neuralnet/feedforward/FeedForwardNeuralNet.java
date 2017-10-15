package matgr.ai.neuralnet.feedforward;

import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronMap;
import matgr.ai.neuralnet.NeuronParameters;
import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.ReadOnlyNeuronMap;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class FeedForwardNeuralNet<NeuronT extends Neuron> {

    private final NeuronFactory<NeuronT> neuronFactory;

    private final NeuronMap<NeuronT> writableInputNeurons;
    private final List<HiddenNeuronLayer<NeuronT>> writableHiddenLayers;

    public final ReadOnlyNeuronMap<NeuronT> inputNeurons;
    public final List<HiddenNeuronLayer<NeuronT>> hiddenLayers;
    public final OutputNeuronLayer<NeuronT> outputLayer;

    public FeedForwardNeuralNet(NeuronFactory<NeuronT> neuronFactory,
                                int inputCount,
                                Iterable<NeuronParameters> outputNeuronsParameters) {

        this(neuronFactory, new OutputNeuronLayer<>(neuronFactory));

        if (null == outputNeuronsParameters) {
            throw new IllegalArgumentException("outputNeuronsParameters not provided");
        }

        for (int i = 0; i < inputCount; i++) {

            NeuronT inputNeuron = neuronFactory.createInput(writableInputNeurons.getNextFreeNeuronId());
            writableInputNeurons.addNeuron(inputNeuron);
        }

        this.outputLayer.setNeurons(inputNeurons.values(), outputNeuronsParameters);
    }

    protected FeedForwardNeuralNet(FeedForwardNeuralNet<NeuronT> other) {

        this(other.neuronFactory, other.outputLayer.deepClone());

        // TODO: can this be relaxed?
        if (other.getClass() != this.getClass()) {
            throw new IllegalArgumentException("Cannot copy neural net of a different type");
        }

        for (NeuronState<NeuronT> neuron : other.writableInputNeurons.values()) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            writableInputNeurons.addNeuron(neuronClone);
        }

        for (HiddenNeuronLayer<NeuronT> layer : other.writableHiddenLayers) {

            HiddenNeuronLayer<NeuronT> layerClone = layer.deepClone();
            writableHiddenLayers.add(layerClone);
        }
    }

    private FeedForwardNeuralNet(NeuronFactory<NeuronT> neuronFactory, OutputNeuronLayer<NeuronT> outputLayer) {

        if (null == neuronFactory) {
            throw new IllegalArgumentException("neuronFactory not provided");
        }

        this.neuronFactory = neuronFactory;

        this.writableInputNeurons = new NeuronMap<>();
        this.inputNeurons = new ReadOnlyNeuronMap<>(this.writableInputNeurons);

        this.writableHiddenLayers = new ArrayList<>();
        this.hiddenLayers = Collections.unmodifiableList(this.writableHiddenLayers);

        this.outputLayer = outputLayer;
    }

    // TODO: do all deepClones like others...
    public FeedForwardNeuralNet deepClone() {
        return new FeedForwardNeuralNet<>(this);
    }

    public int hiddenLayerCount() {
        return writableHiddenLayers.size();
    }

    public void randomizeWeights(RandomGenerator random) {

        for (HiddenNeuronLayer<NeuronT> layer : hiddenLayers) {
            layer.randomizeWeights(random);
        }

        outputLayer.randomizeWeights(random);
    }

    public void addHiddenLayer(Iterable<NeuronParameters> neuronParameters) {

        HiddenNeuronLayer<NeuronT> layer = new HiddenNeuronLayer<>(neuronFactory);

        Iterable<NeuronT> previousLayerNeurons;

        if (hiddenLayers.size() > 0) {
            previousLayerNeurons = hiddenLayers.get(hiddenLayers.size() - 1).neurons.values();
        } else {
            previousLayerNeurons = inputNeurons.values();
        }

        layer.setNeurons(previousLayerNeurons, outputLayer, neuronParameters);

        writableHiddenLayers.add(layer);
    }

    public void setHiddenLayerNeurons(int index, Iterable<NeuronParameters> neuronParameters) {

        if ((index < 0) || (index >= hiddenLayers.size())) {
            throw new IllegalArgumentException("index is out of range");
        }

        HiddenNeuronLayer<NeuronT> layer = writableHiddenLayers.get(index);

        Iterable<NeuronT> previousLayerNeurons;

        if (index == 0) {
            previousLayerNeurons = inputNeurons.values();
        } else {
            previousLayerNeurons = writableHiddenLayers.get(index - 1).neurons.values();
        }

        NeuronLayer<NeuronT> nextLayer;

        if (index == (hiddenLayers.size() - 1)) {
            nextLayer = outputLayer;
        } else {
            nextLayer = writableHiddenLayers.get(index + 1);
        }

        layer.setNeurons(previousLayerNeurons, nextLayer, neuronParameters);
    }

    public void setOutputLayerNeurons(Iterable<NeuronParameters> neuronParameters) {

        Iterable<NeuronT> previousLayerNeurons;

        if (hiddenLayers.size() > 0) {
            previousLayerNeurons = hiddenLayers.get(hiddenLayers.size() - 1).neurons.values();
        } else {
            previousLayerNeurons = inputNeurons.values();
        }

        outputLayer.setNeurons(previousLayerNeurons, neuronParameters);
    }

    public void removeHiddenLayer(int index) {
        // need to clean up connections (reconnect and clear/randomize weights)
        throw new NotImplementedException();
    }

    public List<Double> activate(Collection<Double> inputs, double bias) {

        if (inputNeurons.count() != inputs.size()) {
            throw new IllegalArgumentException("Incorrect number of inputs");
        }

        // initialize inputs
        Iterator<Double> inputIterator = inputs.iterator();
        for (NeuronState<NeuronT> inputNeuron : writableInputNeurons.values()) {

            inputNeuron.postSynapse = inputIterator.next();
        }

        Iterable<NeuronState<NeuronT>> previousNeurons = writableInputNeurons.values();

        for (HiddenNeuronLayer<NeuronT> layer : hiddenLayers) {

            layer.activate(previousNeurons, bias);
            previousNeurons = layer.writableNeurons.values();
        }

        outputLayer.activate(previousNeurons, bias);

        List<Double> outputs = new ArrayList<>();

        for (NeuronState<NeuronT> neuron : outputLayer.writableNeurons()) {
            outputs.add(neuron.postSynapse);
        }

        return outputs;
    }

    public void backPropagate(double learningRate, double bias, List<Double> expectedOutputs) {

        // TODO: handle NaNs

        NeuronLayer<NeuronT> layer = outputLayer;
        Map<Long, Double> nextLayer_dE_dOut = null;

        for (int layerIndex = hiddenLayerCount(); layerIndex >= 0; layerIndex--) {

            HiddenNeuronLayer<NeuronT> previousLayer = null;

            Iterable<NeuronState<NeuronT>> previousNeurons;
            if (layerIndex > 0) {

                previousLayer = writableHiddenLayers.get(layerIndex - 1);
                previousNeurons = previousLayer.writableNeurons.values();

            } else {

                previousNeurons = writableInputNeurons.values();
            }

            Map<Long, Double> thisLayer_dE_dOut = new HashMap<>();

            if (layerIndex == hiddenLayerCount()) {

                for (int neuronIdx = 0; neuronIdx < outputLayer.neuronCount(); neuronIdx++) {

                    NeuronState<NeuronT> neuron = outputLayer.writableNeurons.get(neuronIdx);
                    double neuronOutput = neuron.postSynapse;

                    double neuronExpectedOutput = expectedOutputs.get(neuronIdx);
                    double dE_dOut = -(neuronExpectedOutput - neuronOutput);

                    thisLayer_dE_dOut.put(neuron.neuron.id, dE_dOut);
                }

            } else {

                thisLayer_dE_dOut = nextLayer_dE_dOut;
            }

            nextLayer_dE_dOut = layer.backPropagate(learningRate, bias, previousNeurons, thisLayer_dE_dOut);
            layer = previousLayer;
        }
    }

    public List<Double> getCurrentOutputs() {

        List<Double> outputs = new ArrayList<>();

        for (NeuronState<NeuronT> outputNeuron : outputLayer.writableNeurons()) {

            double output = outputNeuron.postSynapse;
            outputs.add(output);
        }

        return outputs;
    }

    public double getCurrentError(Collection<Double> expectedOutputs) {

        if (outputLayer.neuronCount() != expectedOutputs.size()) {
            throw new IllegalArgumentException("Incorrect number of expected outputs");
        }

        Iterator<Double> expectedOutputIterator = expectedOutputs.iterator();

        double error = 0.0;

        for (NeuronState<NeuronT> outputNeuron : outputLayer.writableNeurons()) {

            double output = outputNeuron.postSynapse;
            double expectedOutput = expectedOutputIterator.next();

            double difference = expectedOutput - output;
            double outputError = (difference * difference);

            error += (0.5 * outputError);
        }

        return error;
    }
}