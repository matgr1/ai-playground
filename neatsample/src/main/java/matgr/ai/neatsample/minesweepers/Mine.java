package matgr.ai.neatsample.minesweepers;


import matgr.ai.neatsample.Point;

public class Mine {

    public final Point location;

    public final long creationIteration;

    public Mine(Point location, long creationIteration) {
        this.location = location;
        this.creationIteration = creationIteration;
    }

    public boolean isActive(long currentIteration, long gestationPeriod) {
        Long iterationsUntilActive = getIterationsUntilActive(currentIteration, gestationPeriod);

        if (null == iterationsUntilActive) {
            return false;
        }

        return iterationsUntilActive <= 0;
    }

    public Long getIterationsUntilActive(long currentIteration, long gestationPeriod) {
        Long activeIteration = getActiveIteration(gestationPeriod);

        if (null == activeIteration) {
            return null;
        }

        long iterationsTilActive = activeIteration - currentIteration;

        if (iterationsTilActive <= 0) {
            return 0L;
        }

        return iterationsTilActive;
    }

    public Long getActiveIteration(long gestationPeriod) {
        if (gestationPeriod < 0) {
            return null;
        }

        return creationIteration + gestationPeriod;
    }
}
