package matgr.ai.math;

import java.util.Comparator;

public abstract class SortComparator<S, T extends Comparable<T>> implements Comparator<S> {

    public final SortDirection direction;

    public final boolean isAscending;
    public final boolean isDescending;

    public SortComparator(SortDirection direction) {
        this.direction = direction;

        this.isAscending = direction == SortDirection.Ascending;
        this.isDescending = !this.isAscending;
    }

    @Override
    public int compare(S x, S y) {

        T comparableX = getComparable(x);
        T comparableY = getComparable(y);

        if (isAscending) {
            return compareItems(comparableX, comparableY);
        }

        return compareItems(comparableY, comparableX);
    }

    protected abstract T getComparable(S s);

    protected abstract int compareItems(T x, T y);

}
