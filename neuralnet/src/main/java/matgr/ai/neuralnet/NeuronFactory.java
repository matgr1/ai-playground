package matgr.ai.neuralnet;

public interface NeuronFactory<NeuronT extends Neuron> {

    NeuronT createBias();

    NeuronT createInput();

    NeuronT createHidden();

    NeuronT createOutput();

}
