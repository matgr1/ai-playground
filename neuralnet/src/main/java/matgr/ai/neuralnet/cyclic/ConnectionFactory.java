package matgr.ai.neuralnet.cyclic;

public interface ConnectionFactory<ConnectionT extends Connection> {

    ConnectionT createConnection(long sourceNeuronId,
                                 long targetNeuronId,
                                 boolean enabled,
                                 double weight);
}
