package matgr.ai.common;

import java.util.Iterator;
import java.util.List;

public class DefaultSizedIterable<T> implements SizedIterable<T> {

    private List<T> list;

    public DefaultSizedIterable(List<T> list) {

        this.list = list;
    }

    @Override
    public int size() {
        return list.size();
    }

    @Override
    public T get(int index) {
        return list.get(index);
    }

    @Override
    public Iterator<T> iterator() {
        return list.iterator();
    }
}
