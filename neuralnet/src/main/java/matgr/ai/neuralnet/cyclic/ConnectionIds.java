package matgr.ai.neuralnet.cyclic;

public class ConnectionIds {

    public final long sourceId;
    public final long targetId;

    public ConnectionIds(long sourceId, long targetId) {
        this.sourceId = sourceId;
        this.targetId = targetId;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        ConnectionIds that = (ConnectionIds) o;

        if (sourceId != that.sourceId) return false;
        if (targetId != that.targetId) return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (sourceId ^ (sourceId >>> 32));
        result = 31 * result + (int) (targetId ^ (targetId >>> 32));
        return result;
    }
}
