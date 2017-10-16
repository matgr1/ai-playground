package matgr.ai.common;

import java.util.Iterator;
import java.util.List;
import java.util.function.Function;

public class SizedSelectIterable<S, T> implements SizedIterable<T> {

    private final Iterable<S> iterable;

    private final int size;
    private final Function<S, T> selectFunc;
    private final Function<Integer, S> getFunc;

    public SizedSelectIterable(List<S> list, Function<S, T> select) {

        this.iterable = list;

        this.size = list.size();
        this.selectFunc = select;

        this.getFunc = list::get;
    }

    public SizedSelectIterable(SizedIterable<S> iterable, Function<S, T> select) {

        this.iterable = iterable;

        this.size = iterable.size();
        this.selectFunc = select;

        this.getFunc = iterable::get;
    }

    @Override
    public int size() {
        return size;
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
