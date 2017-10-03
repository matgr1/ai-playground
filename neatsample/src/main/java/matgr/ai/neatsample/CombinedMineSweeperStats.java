package matgr.ai.neatsample;

import javafx.beans.property.IntegerProperty;
import javafx.beans.property.LongProperty;
import javafx.beans.property.ReadOnlyIntegerProperty;
import javafx.beans.property.ReadOnlyIntegerWrapper;
import javafx.beans.property.ReadOnlyObjectProperty;
import javafx.beans.property.ReadOnlyObjectWrapper;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleLongProperty;

public class CombinedMineSweeperStats implements Cloneable {

    private ReadOnlyObjectWrapper<MineSweeperStatsItem> totals = new ReadOnlyObjectWrapper<>();
    public final MineSweeperStatsItem getTotals() {
        return totals.get();
    }
    private void setTotals(MineSweeperStatsItem value) {
        totals.set(value);
    }
    public final ReadOnlyObjectProperty<MineSweeperStatsItem> totalsProperty() {
        return totals.getReadOnlyProperty();
    }

    private final ReadOnlyObjectWrapper<MineSweeperStatsItem> currentGeneration = new ReadOnlyObjectWrapper<>();
    public final MineSweeperStatsItem getCurrentGeneration() {
        return currentGeneration.get();
    }
    private void setCurrentGeneration(MineSweeperStatsItem value) {
        currentGeneration.set(value);
    }
    public final ReadOnlyObjectProperty<MineSweeperStatsItem> currentGenerationProperty() {
        return currentGeneration.getReadOnlyProperty();
    }

    private final LongProperty generation = new SimpleLongProperty();
    public final Long getGeneration() {
        return generation.get();
    }
    public final void setGeneration(Long value) {
        generation.set(value);
    }
    public final LongProperty generationProperty() {
        return generation;
    }

    private final LongProperty iteration = new SimpleLongProperty();
    public final Long getIteration() {
        return iteration.get();
    }
    public final void setIteration(Long value) {
        iteration.set(value);
    }
    public final LongProperty iterationProperty() {
        return iteration;
    }

    private final LongProperty generationIteration = new SimpleLongProperty();
    public final Long getGenerationIteration() {
        return generationIteration.get();
    }
    public final void setGenerationIteration(Long value) {
        generationIteration.set(value);
    }
    public final LongProperty generationIterationProperty() {
        return generationIteration;
    }

    private final IntegerProperty speciesCount = new SimpleIntegerProperty();
    public final Integer getSpeciesCount() {
        return speciesCount.get();
    }
    public final void setSpeciesCount(Integer value) {
        speciesCount.set(value);
        setMaxSpeciesIndex(value - 1);
    }
    public final IntegerProperty speciesCountProperty() {
        return speciesCount;
    }

    private final ReadOnlyIntegerWrapper maxSpeciesIndex = new ReadOnlyIntegerWrapper();
    public final Integer getMaxSpeciesIndex() {
        return maxSpeciesIndex.get();
    }
    private void setMaxSpeciesIndex(Integer value) {
        maxSpeciesIndex.set(value);
    }
    public final ReadOnlyIntegerProperty maxSpeciesIndexProperty() {
        return maxSpeciesIndex.getReadOnlyProperty();
    }

    public CombinedMineSweeperStats() {
        this(
                new MineSweeperStatsItem(),
                new MineSweeperStatsItem(),
                0,
                0,
                0,
                0,
                0);
    }

    public CombinedMineSweeperStats(CombinedMineSweeperStats other) {
        this(
                other.getTotals().clone(),
                other.getCurrentGeneration().clone(),
                other.getGeneration(),
                other.getIteration(),
                other.getGenerationIteration(),
                other.getSpeciesCount(),
                other.getMaxSpeciesIndex());
    }

    public CombinedMineSweeperStats(MineSweeperStatsItem totals,
                                    MineSweeperStatsItem currentGeneration,
                                    long generation,
                                    long iteration,
                                    long generationIteration,
                                    int speciesCount,
                                    int maxSpeciesIndex) {

        setTotals(totals);
        setCurrentGeneration(currentGeneration);
        setGeneration(generation);
        setIteration(iteration);
        setGenerationIteration(generationIteration);
        setSpeciesCount(speciesCount);
        setMaxSpeciesIndex(maxSpeciesIndex);
    }

    @Override
    public CombinedMineSweeperStats clone() {
        return new CombinedMineSweeperStats(this);
    }
}
