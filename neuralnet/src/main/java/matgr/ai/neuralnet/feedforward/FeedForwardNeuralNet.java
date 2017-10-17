package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.DefaultSizedIterable;
import matgr.ai.common.SizedIterable;
import matgr.ai.common.SizedSelectIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import org.apache.commons.math3.random.RandomGenerator;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

public class FeedForwardNeuralNet<NeuronT extends Neuron> {

    private final NeuronFactory<NeuronT> neuronFactory;

    private final List<NeuronState<NeuronT>> writableInputNeurons;
    private final List<NeuronLayer<NeuronT>> writableHiddenLayers;

    public final SizedIterable<NeuronT> inputNeurons;
    public final List<NeuronLayer<NeuronT>> hiddenLayers;
    public final FullyConnectedLayer<NeuronT> outputLayer;

    public FeedForwardNeuralNet(NeuronFactory<NeuronT> neuronFactory,
                                int inputCount,
                                int outputCount,
                                LayerActivationFunction outputActivationFunction) {

        this(neuronFactory, new FullyConnectedLayer<>(neuronFactory, outputActivationFunction));

        for (int i = 0; i < inputCount; i++) {

            NeuronT inputNeuron = neuronFactory.createInput();
            NeuronState<NeuronT> inputNeuronState = new NeuronState<>(inputNeuron);

            writableInputNeurons.add(inputNeuronState);
        }

        this.outputLayer.setNeurons(outputCount);
        this.outputLayer.connect(inputNeurons);
    }

    protected FeedForwardNeuralNet(FeedForwardNeuralNet<NeuronT> other) {

        this(other.neuronFactory, NeuronLayer.deepClone(other.outputLayer));

        // TODO: can this be relaxed?
        if (other.getClass() != this.getClass()) {
            throw new IllegalArgumentException("Cannot copy neural net of a different type");
        }

        for (NeuronState<NeuronT> neuron : other.writableInputNeurons) {

            NeuronState<NeuronT> neuronClone = neuron.deepClone();
            NeuronState<NeuronT> neuronCloneState = new NeuronState<>(neuronClone);

            writableInputNeurons.add(neuronCloneState);
        }

        for (NeuronLayer<NeuronT> layer : other.writableHiddenLayers) {

            NeuronLayer<NeuronT> layerClone = layer.deepClone();
            writableHiddenLayers.add(layerClone);
        }
    }

    private FeedForwardNeuralNet(NeuronFactory<NeuronT> neuronFactory, FullyConnectedLayer<NeuronT> outputLayer) {

        if (null == neuronFactory) {
            throw new IllegalArgumentException("neuronFactory not provided");
        }

        this.neuronFactory = neuronFactory;

        this.writableInputNeurons = new ArrayList<>();
        this.inputNeurons = new SizedSelectIterable<>(this.writableInputNeurons, n -> n.neuron);

        this.writableHiddenLayers = new ArrayList<>();
        this.hiddenLayers = Collections.unmodifiableList(this.writableHiddenLayers);

        this.outputLayer = outputLayer;
    }

    public FeedForwardNeuralNet deepClone() {
        return new FeedForwardNeuralNet<>(this);
    }

    public int inputNeuronCount() {
        return writableInputNeurons.size();
    }

    public int hiddenLayerCount() {
        return writableHiddenLayers.size();
    }

    public void randomizeWeights(RandomGenerator random) {

        for (NeuronLayer<NeuronT> layer : hiddenLayers) {
            layer.randomizeWeights(random);
        }

        outputLayer.randomizeWeights(random);
    }

    public void addHiddenLayer(int neuronCount, LayerActivationFunction activationFunction) {

        FullyConnectedLayer<NeuronT> layer = new FullyConnectedLayer<>(neuronFactory, activationFunction);
        layer.setNeurons(neuronCount);

        addNewLayer(layer);
    }

    public void addConvolutionalHiddenLayer(int width,
                                            int height,
                                            int kernelRadiusX,
                                            int kernelRadiusY,
                                            LayerActivationFunction activationFunction) {

        ConvolutionalLayer<NeuronT> layer = new ConvolutionalLayer<>(
                neuronFactory,
                activationFunction,
                width,
                height,
                kernelRadiusX,
                kernelRadiusY);

        addNewLayer(layer);
    }

    private void addNewLayer(NeuronLayer<NeuronT> layer) {

        SizedIterable<NeuronT> previousLayerNeurons;

        if (hiddenLayers.size() > 0) {
            previousLayerNeurons = hiddenLayers.get(hiddenLayers.size() - 1).neurons();
        } else {
            previousLayerNeurons = inputNeurons;
        }

        layer.connect(previousLayerNeurons);
        outputLayer.connect(layer.neurons());

        writableHiddenLayers.add(layer);
    }

    public void removeHiddenLayer(int index) {
        // need to clean up connections (reconnect and clear/randomize weights)
        throw new NotImplementedException();
    }

    public List<Double> activate(Collection<Double> inputs, double bias) {

        if (inputNeuronCount() != inputs.size()) {
            throw new IllegalArgumentException("Incorrect number of inputs");
        }

        // initialize inputs
        Iterator<Double> inputIterator = inputs.iterator();
        for (NeuronState<NeuronT> inputNeuron : writableInputNeurons) {

            inputNeuron.postSynapse = inputIterator.next();
        }

        SizedIterable<NeuronState<NeuronT>> previousNeurons = new DefaultSizedIterable<>(writableInputNeurons);

        for (NeuronLayer<NeuronT> layer : hiddenLayers) {

            layer.activate(previousNeurons, bias);
            previousNeurons = layer.writableNeurons();
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

        // initialize post synapse derivatives...
        // TODO: this could maybe be done in one of the other loops?
        // TODO: maybe do a computation version number and if it's less than current values can be initialized..
        for (NeuronLayer<NeuronT> hiddenLayer : writableHiddenLayers) {

            for (NeuronState<NeuronT> neuron : hiddenLayer.writableNeurons()) {

                neuron.postSynapseErrorDerivative = 0.0;
            }
        }

        for (int neuronIdx = 0; neuronIdx < outputLayer.neuronCount(); neuronIdx++) {

            NeuronState<NeuronT> neuron = outputLayer.writableNeurons().get(neuronIdx);
            double neuronOutput = neuron.postSynapse;

            double neuronExpectedOutput = expectedOutputs.get(neuronIdx);
            double dE_dOut = -(neuronExpectedOutput - neuronOutput);

            neuron.postSynapseErrorDerivative = dE_dOut;
        }

        // run backpropagation...
        NeuronLayer<NeuronT> layer = outputLayer;

        for (int layerIndex = hiddenLayerCount(); layerIndex >= 0; layerIndex--) {

            NeuronLayer<NeuronT> previousLayer = null;

            SizedIterable<NeuronState<NeuronT>> previousNeurons;
            if (layerIndex > 0) {

                previousLayer = writableHiddenLayers.get(layerIndex - 1);
                previousNeurons = previousLayer.writableNeurons();

            } else {

                previousNeurons = new DefaultSizedIterable<>(writableInputNeurons);
            }

            layer.backPropagate(previousNeurons, bias, learningRate);
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