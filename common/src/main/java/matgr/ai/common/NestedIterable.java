package matgr.ai.common;

import java.util.Iterator;
import java.util.function.Function;

public class NestedIterable<S, T> implements Iterable<T> {

    private final Iterable<S> iterable;
    private final Function<S, Iterable<T>> getNestedIterable;

    public NestedIterable(Iterable<S> iterable, Function<S, Iterable<T>> getNestedIterable) {

        this.iterable = iterable;
        this.getNestedIterable = getNestedIterable;
    }

    @Override
    public Iterator<T> iterator() {
        return new NestedIterator<>(iterable, getNestedIterable);
    }
}
