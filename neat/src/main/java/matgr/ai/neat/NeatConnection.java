package matgr.ai.neat;

import matgr.ai.neuralnet.Connection;

public class NeatConnection extends Connection {

    public final long innovationNumber;

    public NeatConnection(long sourceId,
                          long targetId,
                          boolean enabled,
                          double weight,
                          long innovationNumber) {

        super(sourceId, targetId, enabled, weight);

        this.innovationNumber = innovationNumber;
    }

    @Override
    protected NeatConnection deepClone() {
        return new NeatConnection(sourceId, targetId, enabled, weight, innovationNumber);
    }
}