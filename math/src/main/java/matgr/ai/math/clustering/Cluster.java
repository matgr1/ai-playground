package matgr.ai.math.clustering;

import java.util.Collections;
import java.util.List;

public class Cluster<ItemT> {

    public final List<ItemT> items;

    public final int representativeIndex;

    public final ItemT representative;

    public Cluster(List<ItemT> items, ItemT representative) throws IllegalArgumentException {
        this(items, representative, getRepresentativeIndex(items, representative));
    }

    public Cluster(List<ItemT> items, int representativeIndex) throws IllegalArgumentException {
        this(items, getRepresentative(items, representativeIndex), representativeIndex);
    }

    private Cluster(List<ItemT> items, ItemT representative, int representativeIndex) {
        this.items = Collections.unmodifiableList(items);
        this.representative = representative;
        this.representativeIndex = representativeIndex;
    }

    private static <ItemT> int getRepresentativeIndex(List<ItemT> items, ItemT representative)
            throws IllegalArgumentException {

        if (null == items) {
            throw new IllegalArgumentException("items not provided");
        }

        for (int i = 0; i < items.size(); i++) {
            ItemT item = items.get(i);
            if (item == representative) {
                return i;
            }
        }

        throw new IllegalArgumentException("Representative does not exist in the list of items");

    }

    private static <ItemT> ItemT getRepresentative(List<ItemT> items, int representativeIndex)
            throws IllegalArgumentException {

        if (null == items) {
            throw new IllegalArgumentException("items not provided");
        }

        if ((representativeIndex < 0) || (representativeIndex >= items.size())) {
            throw new IllegalArgumentException("Invalid representativeIndex");
        }

        return items.get(representativeIndex);

    }

}
