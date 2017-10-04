package matgr.ai.neat;

import matgr.ai.neuralnet.cyclic.*;

import java.util.SortedMap;
import java.util.TreeMap;

public class NeatNeuralNet extends CyclicNeuralNet<NeatConnection, Neuron> {

    // TODO: do this better?
    private SortedMap<Long, NeatConnection> connectionMap;

    public NeatNeuralNet(int inputCount, Iterable<OutputNeuronParameters> outputNeuronsParameters) {

        super(new DefaultNeuronFactory(), new NeatConnectionFactory(), inputCount, outputNeuronsParameters);
    }

    protected NeatNeuralNet(NeatNeuralNet other) {

        super(other);
    }

    @Override
    protected NeatNeuralNet deepClone() {
        return new NeatNeuralNet(this);
    }

    public NeatConnection addConnection(long sourceId,
                                        long targetId,
                                        boolean enabled,
                                        double weight,
                                        long innovationNumber) {

        NeatConnection connection = new NeatConnection(
                sourceId,
                targetId,
                enabled,
                weight,
                innovationNumber);

        addConnection(connection);

        return connection;
    }

    @Override
    protected void addConnection(NeatConnection connection) {
        super.addConnection(connection);
        connectionMap().put(connection.innovationNumber, connection);
    }

    @Override
    public boolean removeConnection(NeatConnection connection) {
        connectionMap().remove(connection.innovationNumber);
        return super.removeConnection(connection);
    }

    SortedMap<Long, NeatConnection> connectionMap() {
        if (connectionMap == null) {
            connectionMap = new TreeMap<>();
        }

        return connectionMap;
    }
}
