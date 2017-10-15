package matgr.ai.neuralnet;

public class NeuronState<NeuronT extends Neuron> {

    public final NeuronT neuron;

    public double preSynapse;
    public double postSynapse;

    public NeuronState(NeuronState<NeuronT> other) {
        this(Neuron.deepClone(other.neuron), other.preSynapse, other.postSynapse);
    }

    public NeuronState(NeuronT neuron, double preSynapse, double postSynapse) {
        this.neuron = neuron;
        this.preSynapse = preSynapse;
        this.postSynapse = postSynapse;
    }

    public NeuronState<NeuronT> deepClone() {
        return new NeuronState<>(this);
    }
}
