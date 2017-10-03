package matgr.ai.neuralnet.cyclic;

import matgr.ai.common.SelectIterable;

import java.util.Set;

public class ReadOnlyNeuronMap<NeuronT extends Neuron> {

    private NeuronMap<NeuronT> neurons;

    public ReadOnlyNeuronMap(NeuronMap<NeuronT> neurons) {
        this.neurons = neurons;
    }

    public int count() {
        return neurons.count();
    }

    public Set<Long> ids() {
        return neurons.ids();
    }

    public Iterable<NeuronT> values() {
        return new SelectIterable<>(neurons.values(), n -> n.neuron);
    }

    public int count(NeuronType neuronType) {
        return neurons.count(neuronType);
    }

    public Set<Long> ids(NeuronType neuronType) {
        return neurons.ids(neuronType);
    }

    public Iterable<NeuronT> values(NeuronType type) {
        return new SelectIterable<>(neurons.values(type), n -> n.neuron);
    }

    public NeuronT get(long neuronId) {

        NeuronState<NeuronT> neuron = neurons.get(neuronId);

        if (neuron == null) {
            return null;
        }

        return neuron.neuron;
    }

    public NeuronT getSingle(NeuronType type) {

        NeuronState<NeuronT> neuron = neurons.getSingle(type);

        if (neuron == null) {
            return null;
        }

        return neuron.neuron;
    }
}
