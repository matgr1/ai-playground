package matgr.ai.neat;

import matgr.ai.neuralnet.ConnectionFactory;

import java.util.concurrent.atomic.AtomicLong;

public class NeatConnectionFactory implements ConnectionFactory<NeatConnection> {

    private static AtomicLong innovationNumber;

    static {
        innovationNumber = new AtomicLong();
    }

    @Override
    public NeatConnection createConnection(long sourceId,
                                           long targetId,
                                           boolean enabled,
                                           double weight) {
        return new NeatConnection(
                sourceId,
                targetId,
                enabled,
                weight,
                innovationNumber.getAndIncrement());
    }
}
