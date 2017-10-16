package matgr.ai.neuralnet;

public class NeuronState<NeuronT extends Neuron> {

    public final NeuronT neuron;

    public double preSynapse;
    public double postSynapse;

    public double postSynapseErrorDerivative;

    public NeuronState(NeuronState<NeuronT> other) {
        this(Neuron.deepClone(other.neuron), other.preSynapse, other.postSynapse, other.postSynapseErrorDerivative);
    }

    public NeuronState(NeuronT neuron) {
        this(neuron, 0.0, 0.0, 0.0);
    }

    private NeuronState(NeuronT neuron, double preSynapse, double postSynapse, double postSynapseErrorDerivative) {
        this.neuron = neuron;
        this.preSynapse = preSynapse;
        this.postSynapse = postSynapse;
        this.postSynapseErrorDerivative = postSynapseErrorDerivative;
    }

    public NeuronState<NeuronT> deepClone() {
        return new NeuronState<>(this);
    }
}
