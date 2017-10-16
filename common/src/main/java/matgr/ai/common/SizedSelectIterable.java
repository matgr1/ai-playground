package matgr.ai.common;

import java.util.Iterator;
import java.util.List;
import java.util.function.Function;
import java.util.function.Supplier;

public class SizedSelectIterable<S, T> implements SizedIterable<T> {

    private final Iterable<S> iterable;

    private final Supplier<Integer> getSize;
    private final Function<Integer, S> getFunc;

    private final Function<S, T> selectFunc;

    public SizedSelectIterable(List<S> list, Function<S, T> select) {

        this.iterable = list;

        this.getSize = list::size;
        this.getFunc = list::get;

        this.selectFunc = select;
    }

    public SizedSelectIterable(SizedIterable<S> iterable, Function<S, T> select) {

        this.iterable = iterable;

        this.getSize = iterable::size;
        this.getFunc = iterable::get;

        this.selectFunc = select;
    }

    @Override
    public int size() {
        return getSize.get();
    }

    @Override
    public T get(int index) {
        return selectFunc.apply(getFunc.apply(index));
    }

    @Override
    public Iterator<T> iterator() {
        return new SelectIterator<>(iterable, selectFunc);
    }
}
