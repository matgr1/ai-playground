package matgr.ai.math;

public class DiscreteDistributionItemSortComparator<T extends DiscreteDistributionItem> extends SortComparator<T, Double> {

    public DiscreteDistributionItemSortComparator(SortDirection direction) {
        super(direction);
    }

    @Override
    protected Double getComparable(T t) {
        return t.getValue();
    }

    @Override
    protected int compareItems(Double x, Double y) {
        return x.compareTo(y);
    }

}
