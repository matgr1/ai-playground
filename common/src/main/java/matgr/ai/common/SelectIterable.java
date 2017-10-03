package matgr.ai.common;

import java.util.Iterator;
import java.util.function.Function;

public class SelectIterable<S, T> implements Iterable<T> {

    private final Iterable<S> iterable;
    private final Function<S, T> select;

    public SelectIterable(Iterable<S> iterable, Function<S, T> select) {

        this.iterable = iterable;
        this.select = select;
    }

    @Override
    public Iterator<T> iterator() {
        return new SelectIterator<>(iterable, select);
    }
}
