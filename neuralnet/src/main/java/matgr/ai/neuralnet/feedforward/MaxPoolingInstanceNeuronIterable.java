package matgr.ai.neuralnet.feedforward;

import matgr.ai.common.NestedIterator;
import matgr.ai.common.SizedIterable;
import matgr.ai.neuralnet.Neuron;
import matgr.ai.neuralnet.NeuronState;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

class MaxPoolingInstanceNeuronIterable<NeuronT extends Neuron> implements SizedIterable<NeuronState<NeuronT>> {

    private final int width;
    private final int height;
    private final int depth;

    private final MaxPoolingInstance<NeuronT>[] instances;

    private final int volumeSize;
    private final int planeSize;

    public MaxPoolingInstanceNeuronIterable(int width,
                                            int height,
                                            int depth,
                                            MaxPoolingInstance<NeuronT>[] instances) {

        this.width = width;
        this.height = height;
        this.depth = depth;

        this.instances = instances;

        this.planeSize = width * height;
        this.volumeSize = planeSize * depth;
    }

    @Override
    public int size() {
        return instances.length * width * height * depth;
    }

    @Override
    public NeuronState<NeuronT> get(int index) {

        int instance = index / volumeSize;
        int instanceIndex = index % volumeSize;

        int plane = instanceIndex / planeSize;
        int planeIndex = instanceIndex % planeSize;

        int row = planeIndex / width;
        int col = planeIndex % width;

        return instances[instance].neurons[plane][row][col];
    }

    @Override
    @Nonnull
    public Iterator<NeuronState<NeuronT>> iterator() {

        List<MaxPoolingInstance<NeuronT>> instanceList = Arrays.asList(instances);
        return new NestedIterator<>(instanceList, i -> new NeuronArrayIterable<>(width, height, depth, i.neurons));
    }
}