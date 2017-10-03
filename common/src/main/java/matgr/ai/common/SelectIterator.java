package matgr.ai.common;

import java.util.Iterator;
import java.util.function.Function;

public class SelectIterator<S, T> implements Iterator<T> {

    private final Iterator<S> iterator;
    private final Function<S, T> select;

    public SelectIterator(Iterable<S> iterable, Function<S, T> select) {

        this.iterator = iterable.iterator();
        this.select = select;
    }

    @Override
    public boolean hasNext() {
        return iterator.hasNext();
    }

    @Override
    public T next() {
        S value = iterator.next();
        return select.apply(value);
    }
}
