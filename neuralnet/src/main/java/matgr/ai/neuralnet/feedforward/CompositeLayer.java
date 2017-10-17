package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import org.apache.commons.math3.random.RandomGenerator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class CompositeLayer<NeuronT extends Neuron> extends NeuronLayer<NeuronT> {

    private final List<NeuronLayer<NeuronT>> layers;

    @SafeVarargs
    protected CompositeLayer(NeuronFactory<NeuronT> neuronFactory, NeuronLayer<NeuronT>... layers) {

        super(neuronFactory);

        this.layers = new ArrayList<>();
        Collections.addAll(this.layers, layers);

        if (this.layers.size() < 1) {
            throw new IllegalArgumentException("No layers provided");
        }
    }

    protected CompositeLayer(CompositeLayer<NeuronT> other) {
        super(other);

        this.layers = new ArrayList<>();

        for (NeuronLayer<NeuronT> layer : other.layers) {

            this.layers.add(NeuronLayer.deepClone(layer));
        }
    }

    @Override
    protected CompositeLayer<NeuronT> deepClone() {
        return new CompositeLayer<>(this);
    }

    @Override
    public int inputCount() {
        return firstLayer().inputCount();
    }

    @Override
    public int outputCount() {
        return lastLayer().outputCount();
    }

    @Override
    public SizedIterable<NeuronT> outputNeurons() {
        return lastLayer().outputNeurons();
    }

    @Override
    SizedIterable<NeuronState<NeuronT>> outputWritableNeurons() {
        return lastLayer().outputWritableNeurons();
    }

    @Override
    void randomizeWeights(RandomGenerator random) {

        for (NeuronLayer<NeuronT> layer : layers) {

            layer.randomizeWeights(random);
        }
    }

    @Override
    void connect(SizedIterable<NeuronT> previousLayerNeurons) {

        SizedIterable<NeuronT> previousNeurons = previousLayerNeurons;

        for (NeuronLayer<NeuronT> layer : layers) {

            layer.connect(previousNeurons);
            previousNeurons = layer.outputNeurons();
        }
    }

    @Override
    void activate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias) {

        SizedIterable<NeuronState<NeuronT>> previousNeurons = previousLayerNeurons;

        for (NeuronLayer<NeuronT> layer : layers) {

            layer.activate(previousNeurons, bias);
            previousNeurons = layer.outputWritableNeurons();
        }
    }

    @Override
    void resetPostSynapseErrorDerivatives(double value) {

        for (NeuronLayer<NeuronT> layer : layers) {

            layer.resetPostSynapseErrorDerivatives(value);
        }
    }

    @Override
    void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias, double learningRate) {

        NeuronLayer<NeuronT> layer = lastLayer();

        for (int layerIndex = layers.size() - 1; layerIndex >= 0; layerIndex--) {

            NeuronLayer<NeuronT> previousLayer = null;

            SizedIterable<NeuronState<NeuronT>> previousNeurons;
            if (layerIndex > 0) {

                previousLayer = layers.get(layerIndex - 1);
                previousNeurons = previousLayer.outputWritableNeurons();

            } else {

                previousNeurons = previousLayerNeurons;
            }

            layer.backPropagate(previousNeurons, bias, learningRate);
            layer = previousLayer;
        }
    }

    private NeuronLayer<NeuronT> firstLayer() {
        return layers.get(0);
    }

    private NeuronLayer<NeuronT> lastLayer() {
        return layers.get(layers.size() - 1);
    }
}
