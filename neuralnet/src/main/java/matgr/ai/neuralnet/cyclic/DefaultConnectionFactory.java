package matgr.ai.neuralnet.cyclic;

import matgr.ai.neuralnet.cyclic.Connection;
import matgr.ai.neuralnet.cyclic.ConnectionFactory;

public class DefaultConnectionFactory implements ConnectionFactory<Connection> {

    @Override
    public Connection createConnection(long sourceId,
                                       long targetId,
                                       boolean enabled,
                                       double weight) {

        return new Connection(sourceId, targetId, enabled, weight);
    }
}
