package matgr.ai.genetic.selection;

public abstract class RankBasedSelectionStrategy extends SelectionStrategy {

    public final double selectivePressure;

    protected RankBasedSelectionStrategy(double selectivePressure) {

        if ((selectivePressure < 0.0) || (selectivePressure > 1.0)) {
            throw new IllegalArgumentException("Selective pressure must be in the range [0.0, 1.0]");
        }

        this.selectivePressure = selectivePressure;

    }

    // TODO: allow this still?
    @Override
    public boolean useGroupedSelectionSampling() {
        return false;
    }

}
