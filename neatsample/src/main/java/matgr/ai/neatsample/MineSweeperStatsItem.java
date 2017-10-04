package matgr.ai.neatsample;

import javafx.beans.property.*;

public class MineSweeperStatsItem implements Cloneable {

    private IntegerProperty explosions = new SimpleIntegerProperty();
    public Integer getExplosions() {
        return explosions.get();
    }
    public void setExplosions(Integer value) {
        explosions.set(value);
        setExplosionsToClearedRatio((double)value / (double)getCleared());
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
        setExplosionsToClearedRatio((double)getExplosions() / (double)value);
    }
    public IntegerProperty clearedProperty() {
        return cleared;
    }

    private ReadOnlyDoubleWrapper explosionsToClearedRatio = new ReadOnlyDoubleWrapper();
    public Double getExplosionsToClearedRatio() {
        return explosionsToClearedRatio.get();
    }
    private void setExplosionsToClearedRatio(Double value) {
        explosionsToClearedRatio.set(value);
        setExplosionsToClearedRatioDisplay(String.format("%.2f", value));
    }
    public ReadOnlyDoubleProperty explosionsToClearedRatioProperty() {
        return explosionsToClearedRatio.getReadOnlyProperty();
    }

    private ReadOnlyStringWrapper explosionsToClearedRatioDisplay = new ReadOnlyStringWrapper();
    public String getExplosionsToClearedRatioDisplay() {
        return explosionsToClearedRatioDisplay.get();
    }
    private void setExplosionsToClearedRatioDisplay(String value) {
        explosionsToClearedRatioDisplay.set(value);
    }
    public ReadOnlyStringProperty explosionsToClearedRatioDisplayProperty() {
        return explosionsToClearedRatioDisplay.getReadOnlyProperty();
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
        setFitnessDisplay(String.format("%.2f", value));
    }
    public DoubleProperty fitnessProperty() {
        return fitness;
    }

    private ReadOnlyStringWrapper fitnessDisplay = new ReadOnlyStringWrapper();
    public String getFitnessDisplay() {
        return fitnessDisplay.get();
    }
    private void setFitnessDisplay(String value) {
        fitnessDisplay.set(value);
    }
    public ReadOnlyStringProperty fitnessDisplayProperty() {
        return fitnessDisplay.getReadOnlyProperty();
    }

    public MineSweeperStatsItem(){
        this(0, 0, 0, 0, 0);
    }

    public MineSweeperStatsItem(MineSweeperStatsItem other){
        this(
                other.getExplosions(),
                other.getCleared(),
                other.getExplosionsPerIteration(),
                other.getMinesClearedPerIteration(),
                other.getFitness());
    }

    public MineSweeperStatsItem(int explosions,
                                int cleared,
                                double explosionsPerIteration,
                                double minesClearedPerIteration,
                                double fitness){
        setExplosions(explosions);
        setCleared(cleared);
        setExplosionsPerIteration(explosionsPerIteration);
        setMinesClearedPerIteration(minesClearedPerIteration);
        setFitness(fitness);
    }

    @Override
    public MineSweeperStatsItem clone(){
        return new MineSweeperStatsItem(this);
    }
}
