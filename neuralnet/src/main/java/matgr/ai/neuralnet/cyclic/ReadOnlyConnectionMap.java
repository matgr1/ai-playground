package matgr.ai.neuralnet.cyclic;

import java.util.List;
import java.util.Set;

public class ReadOnlyConnectionMap<ConnectionT extends Connection> {

    private final ConnectionMap<ConnectionT> connections;

    public ReadOnlyConnectionMap(ConnectionMap<ConnectionT> connections) {
        this.connections = connections;
    }

    public Iterable<ConnectionT> values() {
        return connections.values();
    }

    public Set<Long> sourceIds() {
        return connections.sourceIds();
    }

    public Set<Long> targetIds(long sourceId) {
        return connections.targetIds(sourceId);
    }

    public boolean isConnected(long sourceId, long targetId) {
        return connections.isConnected(sourceId, targetId);
    }

    public ConnectionT getConnection(long sourceId, long targetId) {
        return connections.getConnection(sourceId, targetId);
    }

    public int count() {
        return connections.count();
    }

    public int getIncomingConnectionCount(long neuronId) {
        return connections.getIncomingConnectionCount(neuronId);
    }

    public int getOutgoingConnectionCount(long neuronId) {
        return connections.getOutgoingConnectionCount(neuronId);
    }

    public List<ConnectionT> getIncomingConnections(long neuronId) {
        return connections.getIncomingConnections(neuronId);
    }

    public List<ConnectionT> getOutgoingConnections(long neuronId) {
        return connections.getOutgoingConnections(neuronId);
    }
}
