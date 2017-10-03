package matgr.ai.neat;

import java.util.Iterator;
import java.util.SortedMap;

// TODO: there's probably a better way of doing this...

class SortedConnectionGeneIterator {

    private final Iterator<NeatConnection> sortedIterator;

    private NeatConnection previous;
    private NeatConnection current;

    private boolean atStart;
    private boolean pastEnd;

    public SortedConnectionGeneIterator(SortedMap<Long, NeatConnection> connectionMap) {
        this(connectionMap.values().iterator());
    }

    public SortedConnectionGeneIterator(Iterator<NeatConnection> sortedIterator) {
        this.sortedIterator = sortedIterator;
        increment();
    }

    public NeatConnection getPrevious() {
        return previous;
    }

    public NeatConnection getCurrent() {
        return current;
    }

    public boolean isAtStart() {
        return atStart;
    }

    public boolean isPastEnd() {
        return pastEnd;
    }

    public void increment() {

        if (pastEnd) {
            throw new IllegalStateException("Attempt to increment past end");
        }

        previous = current;
        atStart = (null == previous);

        if (sortedIterator.hasNext()) {
            current = sortedIterator.next();
        } else {
            pastEnd = true;
        }

    }

}
