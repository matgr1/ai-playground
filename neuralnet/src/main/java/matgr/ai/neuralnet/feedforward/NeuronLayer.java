package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.SizedIterable;
import matgr.ai.math.RandomFunctions;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronFactory;
import matgr.ai.neuralnet.NeuronState;
import org.apache.commons.math3.random.RandomGenerator;

public abstract class NeuronLayer<NeuronT extends Neuron> {

    protected final NeuronFactory<NeuronT> neuronFactory;

    protected NeuronLayer(NeuronFactory<NeuronT> neuronFactory) {

        this.neuronFactory = neuronFactory;
    }

    protected NeuronLayer(NeuronLayer<NeuronT> other) {

        if (null == other) {
            throw new IllegalArgumentException("other");
        }

        this.neuronFactory = other.neuronFactory;
    }

    protected abstract NeuronLayer<NeuronT> deepClone();

    public static <NeuronLayerT extends NeuronLayer> NeuronLayerT deepClone(NeuronLayerT layer) {

        @SuppressWarnings("unchecked")
        NeuronLayerT clone = (NeuronLayerT) layer.deepClone();

        if (clone.getClass() != layer.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    public abstract int neuronCount();

    public abstract SizedIterable<NeuronT> neurons();

    abstract SizedIterable<NeuronState<NeuronT>> writableNeurons();

    abstract void randomizeWeights(RandomGenerator random);

    abstract void connect(SizedIterable<NeuronT> previousLayerNeurons);

    abstract void activate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons, double bias);

    abstract void backPropagate(SizedIterable<NeuronState<NeuronT>> previousLayerNeurons,
                                double bias,
                                double learningRate);

    protected static double getRandomWeight(RandomGenerator random) {
        return RandomFunctions.nextDouble(random, -1.0, 1.0);
    }
}
