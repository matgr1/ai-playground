package matgr.ai.neuralnet;

public class DefaultNeuronFactory implements NeuronFactory<Neuron> {

    @Override
    public Neuron createBias() {
        return Neuron.bias();
    }

    @Override
    public Neuron createInput() {
        return Neuron.input();
    }

    @Override
    public Neuron createHidden() {
        return Neuron.hidden();
    }

    @Override
    public Neuron createOutput() {
        return Neuron.output();
    }
}
