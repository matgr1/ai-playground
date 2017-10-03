package matgr.ai.neuralnet.cyclic;

public class Connection {

    public final long sourceId;
    public final long targetId;

    public boolean enabled;
    public double weight;

    public Connection(long sourceId, long targetId, boolean enabled, double weight) {

        this.sourceId = sourceId;
        this.targetId = targetId;

        this.enabled = enabled;
        this.weight = weight;
    }

    public static <ConnectionT extends Connection> ConnectionT deepClone(ConnectionT connection) {

        @SuppressWarnings("unchecked")
        ConnectionT clone = (ConnectionT) connection.deepClone();

        if (clone.getClass() != connection.getClass()) {
            throw new IllegalArgumentException("Invalid item - clone not overridden correctly in derived class");
        }

        return clone;
    }

    protected Connection deepClone() {
        return new Connection(sourceId, targetId, enabled, weight);
    }
}
