package matgr.ai.neuralnet.cyclic;

import matgr.ai.common.NestedIterator;
import matgr.ai.neuralnet.Connection;

import javax.annotation.Nonnull;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

class ConnectionMap<ConnectionT extends Connection> {

    private final Map<Long, Map<Long, ConnectionT>> connections;

    private final Map<Long, List<ConnectionT>> incomingConnections;
    private final Map<Long, List<ConnectionT>> outgoingConnections;

    private int connectionCount;

    public ConnectionMap() {

        connectionCount = 0;

        connections = new HashMap<>();
        incomingConnections = new HashMap<>();
        outgoingConnections = new HashMap<>();
    }

    public Iterable<ConnectionT> values() {
        return new ConnectionMapIterable(connections);
    }

    public Set<Long> sourceIds() {
        return connections.keySet();
    }

    public Set<Long> targetIds(long sourceId) {
        Map<Long, ConnectionT> targetMap = connections.get(sourceId);

        if (targetMap == null) {
            return new HashSet<>();
        }

        return targetMap.keySet();
    }

    public boolean isConnected(long sourceId, long targetId) {

        Map<Long, ConnectionT> targetConnectionMap = connections.get(sourceId);

        if (targetConnectionMap != null) {

            ConnectionT connection = targetConnectionMap.get(targetId);

            if (connection != null) {
                return true;
            }
        }

        return false;
    }

    public ConnectionT getConnection(long sourceId, long targetId) {

        Map<Long, ConnectionT> targetConnectionMap = connections.get(sourceId);

        if (targetConnectionMap != null) {

            return targetConnectionMap.get(targetId);
        }

        return null;
    }

    public int count() {
        return connectionCount;
    }

    public int getIncomingConnectionCount(long neuronId) {

        int count = 0;

        List<ConnectionT> incoming = incomingConnections.get(neuronId);
        if (incoming != null) {
            count += incoming.size();
        }

        return count;
    }

    public int getOutgoingConnectionCount(long neuronId) {

        int count = 0;

        List<ConnectionT> outgoing = outgoingConnections.get(neuronId);
        if (outgoing != null) {
            count += outgoing.size();
        }

        return count;
    }

    public List<ConnectionT> getIncomingConnections(long neuronId) {

        List<ConnectionT> incoming = incomingConnections.get(neuronId);

        if (incoming == null) {
            incoming = new ArrayList<>();
        }

        return Collections.unmodifiableList(incoming);
    }

    public List<ConnectionT> getOutgoingConnections(long neuronId) {

        List<ConnectionT> outgoing = outgoingConnections.get(neuronId);

        if (outgoing == null) {
            outgoing = new ArrayList<>();
        }

        return Collections.unmodifiableList(outgoing);
    }

    public void addConnection(ConnectionT connection) {

        Map<Long, ConnectionT> targetMap = connections.computeIfAbsent(connection.sourceId, k -> new HashMap<>());

        ConnectionT existing = targetMap.get(connection.targetId);

        if (existing != null) {
            throw new IllegalStateException(
                    String.format(
                            "The connection %s-->%s is already present",
                            connection.sourceId,
                            connection.targetId));
        }

        List<ConnectionT> incoming = incomingConnections.computeIfAbsent(connection.targetId, k -> new ArrayList<>());
        List<ConnectionT> outgoing = outgoingConnections.computeIfAbsent(connection.sourceId, k -> new ArrayList<>());

        targetMap.put(connection.targetId, connection);

        incoming.add(connection);
        outgoing.add(connection);

        connectionCount++;
    }

    public boolean removeConnection(ConnectionT connection) {

        Map<Long, ConnectionT> targetMap = connections.get(connection.sourceId);

        if (targetMap != null) {

            connection = targetMap.get(connection.targetId);

            if (connection != null) {

                targetMap.remove(connection.targetId);

                if (targetMap.size() < 1) {
                    connections.remove(connection.sourceId);
                }

                ConnectionT conn = connection;

                List<ConnectionT> incoming = incomingConnections.get(connection.targetId);

                if (!incoming.removeIf(c -> (c.sourceId == conn.sourceId) && (c.targetId == conn.targetId))) {
                    throw new AssertionError("Failed to remove connection");
                }

                if (incoming.size() < 1) {
                    incomingConnections.remove(connection.targetId);
                }

                List<ConnectionT> outgoing = outgoingConnections.get(connection.sourceId);

                if (!outgoing.removeIf(c -> (c.sourceId == conn.sourceId) && (c.targetId == conn.targetId))) {
                    throw new AssertionError("Failed to remove connection");
                }

                if (outgoing.size() < 1) {
                    outgoingConnections.remove(connection.sourceId);
                }

                connectionCount--;
                return true;
            }
        }

        return false;
    }

    private class ConnectionMapIterable implements Iterable<ConnectionT> {

        private final Map<Long, Map<Long, ConnectionT>> connections;

        public ConnectionMapIterable(Map<Long, Map<Long, ConnectionT>> connections) {
            this.connections = connections;
        }

        @Override
        @Nonnull
        public Iterator<ConnectionT> iterator() {
            return new NestedIterator<>(connections.values(), Map::values);
        }
    }
}
