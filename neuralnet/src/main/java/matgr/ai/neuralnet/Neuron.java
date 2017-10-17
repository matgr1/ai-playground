package matgr.ai.neuralnet;

public class Neuron {

    public final NeuronType type;

    protected Neuron(NeuronType type) {

        this.type = type;
    }

    public static Neuron bias() {
        return new Neuron(NeuronType.Bias);
    }

    public static Neuron input() {
        return new Neuron(NeuronType.Input);
    }

    public static Neuron hidden() {

        return new Neuron(NeuronType.Hidden);
    }

    public static Neuron output() {

        return new Neuron(NeuronType.Output);
    }

    public static <NeuronT extends Neuron> NeuronT deepClone(NeuronT neuron) {

        @SuppressWarnings("unchecked")
        NeuronT clone = (NeuronT) neuron.deepClone();

        if (clone.getClass() != neuron.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    protected Neuron deepClone() {
        return new Neuron(type);
    }
}