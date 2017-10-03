package matgr.ai.neatsample;

import javafx.beans.property.DoubleProperty;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.SimpleDoubleProperty;
import javafx.beans.property.SimpleIntegerProperty;

public class MineSweeperStatsItem {

    private IntegerProperty explosions = new SimpleIntegerProperty();
    public Integer getExplosions() {
        return explosions.get();
    }
    public void setExplosions(Integer value) {
        explosions.set(value);
    }
    public IntegerProperty explosionsProperty() {
        return explosions;
    }

    private IntegerProperty cleared = new SimpleIntegerProperty();
    public Integer getCleared() {
        return cleared.get();
    }
    public void setCleared(Integer value) {
        cleared.set(value);
    }
    public IntegerProperty clearedProperty() {
        return cleared;
    }

    private DoubleProperty explosionsPerIteration = new SimpleDoubleProperty();
    public Double getExplosionsPerIteration() {
        return explosionsPerIteration.get();
    }
    public void setExplosionsPerIteration(Double value) {
        explosionsPerIteration.set(value);
    }
    public DoubleProperty explosionsPerIterationProperty() {
        return explosionsPerIteration;
    }

    private DoubleProperty minesClearedPerIteration = new SimpleDoubleProperty();
    public Double getMinesClearedPerIteration() {
        return minesClearedPerIteration.get();
    }
    public void setMinesClearedPerIteration(Double value) {
        minesClearedPerIteration.set(value);
    }
    public DoubleProperty minesClearedPerIterationProperty() {
        return minesClearedPerIteration;
    }

    private DoubleProperty fitness = new SimpleDoubleProperty();
    public Double getFitness() {
        return fitness.get();
    }
    public void setFitness(Double value) {
        fitness.set(value);
    }
    public DoubleProperty fitnessProperty() {
        return fitness;
    }
}
