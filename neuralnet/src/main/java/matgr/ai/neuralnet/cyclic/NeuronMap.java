package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.NeuronState;
import matgr.ai.neuralnet.NeuronType;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

class NeuronMap<NeuronT extends CyclicNeuron> {

    private SortedMap<Long, NeuronState<NeuronT>> neuronMap;
    private Map<NeuronType, SortedMap<Long, NeuronState<NeuronT>>> neuronTypeMap;

    public NeuronMap() {

        neuronMap = new TreeMap<>();
        neuronTypeMap = new HashMap<>();

        for (NeuronType type : NeuronType.values()) {
            neuronTypeMap.put(type, new TreeMap<>());
        }
    }

    public int count() {
        return neuronMap.size();
    }

    public Set<Long> ids() {
        return neuronMap.keySet();
    }

    public Iterable<NeuronState<NeuronT>> values() {
        return neuronMap.values();
    }

    public int count(NeuronType type) {
        return neuronTypeMap.get(type).size();
    }

    public Set<Long> ids(NeuronType type) {
        return neuronTypeMap.get(type).keySet();
    }

    public Iterable<NeuronState<NeuronT>> values(NeuronType type) {
        return neuronTypeMap.get(type).values();
    }

    public NeuronState<NeuronT> get(long neuronId) {
        return neuronMap.get(neuronId);
    }

    public NeuronState<NeuronT> getSingle(NeuronType type) {

        SortedMap<Long, NeuronState<NeuronT>> map = neuronTypeMap.get(type);

        if (map == null) {
            return null;
        }

        if (map.size() == 0) {
            return null;
        }

        if (map.size() > 1) {
            throw new IllegalArgumentException(
                    String.format("The type %s has more than one entry", type.name()));
        }

        return map.get(map.lastKey());
    }

    public long getNextFreeNeuronId() {

        if (neuronMap.size() < 1) {
            return 0;
        }

        return neuronMap.lastKey() + 1;
    }

    public void addNeuron(NeuronT neuron) {
        addNeuron(new NeuronState<>(neuron));
    }

    public void addNeuron(NeuronState<NeuronT> neuron) {

        if (neuronMap.containsKey(neuron.neuron.id)) {
            throw new IllegalArgumentException("Neuron already added");
        }

        if (neuron.neuron.type == NeuronType.Bias) {
            SortedMap<Long, NeuronState<NeuronT>> neurons = neuronTypeMap.get(neuron.neuron.type);
            neurons.clear();
        }

        neuronMap.put(neuron.neuron.id, neuron);
        neuronTypeMap.get(neuron.neuron.type).put(neuron.neuron.id, neuron);
    }

    public boolean removeNeuron(long neuronId) {

        NeuronState<NeuronT> neuron = neuronMap.get(neuronId);

        if (neuron == null) {
            return false;
        }

        SortedMap<Long, NeuronState<NeuronT>> map = neuronTypeMap.get(neuron.neuron.type);

        neuronMap.remove(neuronId);
        map.remove(neuronId);

        return true;
    }
}
