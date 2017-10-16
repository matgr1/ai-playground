package matgr.ai.common;

public interface SizedIterable<T> extends Iterable<T> {

    int size();

    T get(int index);
}
