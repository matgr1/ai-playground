package matgr.ai.common;

import java.util.Iterator;
import java.util.function.Function;

public class NestedIterator<S, T> implements Iterator<T> {

    private final Iterator<S> outerIterator;
    private final Function<S, Iterable<T>> getNestedIterable;

    private Iterator<T> currentNestedIterator;

    public NestedIterator(Iterable<S> iterable, Function<S, Iterable<T>> getNestedIterable) {

        this.outerIterator = iterable.iterator();
        this.getNestedIterable = getNestedIterable;
    }

    @Override
    public boolean hasNext() {

        if (currentNestedIterator == null) {

            currentNestedIterator = getNextNonEmptyNestedIterator();

        } else {

            if (!currentNestedIterator.hasNext()) {
                currentNestedIterator = getNextNonEmptyNestedIterator();
            }

        }

        if (currentNestedIterator == null) {
            return false;
        }

        return true;
    }

    @Override
    public T next() {
        return currentNestedIterator.next();
    }

    private Iterator<T> getNextNonEmptyNestedIterator() {

        while (outerIterator.hasNext()) {

            S nextOuter = outerIterator.next();
            Iterator<T> nextNested = getNestedIterable.apply(nextOuter).iterator();

            if (nextNested.hasNext()) {
                return nextNested;
            }

        }

        return null;

    }
}
