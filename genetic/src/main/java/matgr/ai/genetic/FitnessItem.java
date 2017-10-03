package matgr.ai.genetic;

import matgr.ai.math.DiscreteDistributionItem;

public class FitnessItem<T> implements DiscreteDistributionItem {

    public final T item;

    public final double fitness;

    public FitnessItem(T item, double fitness) {

        if (null == item) {
            throw new IllegalArgumentException("item not provided");
        }

        this.item = item;
        this.fitness = fitness;

    }

    @Override
    public double getValue() {
        return fitness;
    }
}