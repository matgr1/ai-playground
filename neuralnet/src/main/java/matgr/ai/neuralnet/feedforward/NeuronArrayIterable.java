package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.NestedIterable;
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
    private final int depth;

    private final NeuronState<NeuronT>[][][] neurons;

    public NeuronArrayIterable(int width, int height, int depth, NeuronState<NeuronT>[][][] neurons) {
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.neurons = neurons;
    }

    @Override
    public int size() {
        return width * height * depth;
    }

    @Override
    public NeuronState<NeuronT> get(int index) {

        int plane = index / (width * height);
        int planeIndex = index % (width * height);

        int row = planeIndex / width;
        int col = planeIndex % width;

        return neurons[plane][row][col];
    }

    @Override
    @Nonnull
    public Iterator<NeuronState<NeuronT>> iterator() {

        List<NeuronState<NeuronT>[][]> outerList = Arrays.asList(neurons);

        NestedIterable<NeuronState<NeuronT>[][], NeuronState<NeuronT>[]> outerIerable =
                new NestedIterable<>(outerList, Arrays::asList);

        return new NestedIterator<>(outerIerable, Arrays::asList);
    }
}
