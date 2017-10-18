package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.NestedIterator;
import matgr.ai.common.SizedIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronState;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

class NeuronArrayIterable<NeuronT extends Neuron> implements SizedIterable<NeuronState<NeuronT>> {

    private final int width;
    private final int height;

    private final NeuronState<NeuronT>[][] neurons;

    public NeuronArrayIterable(int width, int height, NeuronState<NeuronT>[][] neurons) {
        this.width = width;
        this.height = height;
        this.neurons = neurons;
    }

    @Override
    public int size() {
        return width * height;
    }

    @Override
    public NeuronState<NeuronT> get(int index) {

        int row = index / width;
        int col = index % width;

        return neurons[row][col];
    }

    @Override
    @Nonnull
    public Iterator<NeuronState<NeuronT>> iterator() {

        List<NeuronState<NeuronT>[]> neuronsList = Arrays.asList(neurons);
        return new NestedIterator<>(neuronsList, Arrays::asList);
    }
}
