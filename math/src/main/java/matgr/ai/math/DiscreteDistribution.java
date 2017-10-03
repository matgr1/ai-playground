package matgr.ai.math;

import com.google.common.collect.ObjectArrays;
import org.apache.commons.math3.random.RandomGenerator;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;

public class DiscreteDistribution<T> {

    private final List<ValueItem<T>> items;
    private final List<ValueGroup<T>> groupedItems;

    public final DistributionStats stats;
    public final DistributionStats groupedStats;

    private DiscreteDistribution(List<ValueItem<T>> items,
                                 DistributionStats stats,
                                 List<ValueGroup<T>> groupedItems,
                                 DistributionStats groupedStats) {
        this.items = items;
        this.stats = stats;

        this.groupedItems = groupedItems;
        this.groupedStats = groupedStats;
    }

    public static <T> DiscreteDistribution<T>[] createArray(int count) {
        @SuppressWarnings("unchecked")
        DiscreteDistribution<T>[] result = (DiscreteDistribution<T>[]) Array.newInstance(DiscreteDistribution.class, count);
        return result;
    }

    public static <T> DiscreteDistribution<T> create(Iterable<T> items, Map<T, Double> values) {
        return create(items, values::get);
    }

    public static <T extends DiscreteDistributionItem> DiscreteDistribution<T> create(Iterable<T> items) {
        return create(items, DiscreteDistributionItem::getValue);
    }

    public DiscreteDistribution<T> removeOutcome(T item) {
        return removeOutcome(item, null);
    }

    public DiscreteDistribution<T> removeOutcome(T item, BiFunction<T, T, Boolean> equalityOverride) {

        RemoveOutcomeResult<ValueItem<T>> removeItemResult = removeOutcomeItem(items, item, equalityOverride);
        RemoveOutcomeResult<ValueGroup<T>> removeGroupResult = removeOutcomeGroup(groupedItems, item, equalityOverride);

        return new DiscreteDistribution<>(
                removeItemResult.result,
                removeItemResult.stats,
                removeGroupResult.result,
                removeGroupResult.stats);
    }

    public T sample(RandomGenerator random, boolean grouped) {

        if (grouped) {
            return sampleGroups(random, groupedItems, groupedStats);
        }

        return sampleItems(random, items, stats);
    }

    private static <T> T sampleItems(RandomGenerator random, List<ValueItem<T>> items, DistributionStats stats) {
        ValueItem<T> item = stochasticAcceptance(random, items, stats);

        if (null == item) {
            throw new IllegalStateException("Distribution is empty");
        }

        return item.item;
    }

    private static <T> T sampleGroups(RandomGenerator random,
                                      List<ValueGroup<T>> groupedItems,
                                      DistributionStats groupedStats) {

        ValueGroup<T> group = stochasticAcceptance(random, groupedItems, groupedStats);

        if (null == group) {
            throw new IllegalStateException("Distribution is empty");
        }

        if (group.items.size() == 1) {
            return group.items.get(0);
        }

        T item = RandomFunctions.selectItem(random, group.items, null);

        if (item == null) {
            throw new IllegalStateException("Invalid group");
        }

        return item;

    }

    private static <S extends DiscreteDistributionItem> S stochasticAcceptance(RandomGenerator random,
                                                                               List<S> items,
                                                                               DistributionStats stats) {

        if (items.size() < 1) {
            return null;
        }

        if (stats.min < 0) {
            throw new IllegalStateException("Values must all be greater than or equal to 0");
        }

        if (stats.max == stats.min) {
            return RandomFunctions.selectItem(random, items);
        }

        S acceptedItem = null;

        while (null == acceptedItem) {
            S testItem = RandomFunctions.selectItem(random, items);

            double normalizedValue = testItem.getValue();

            double testValue = RandomFunctions.nextDouble(random, 0.0, stats.max);

            if (testValue < normalizedValue) {
                acceptedItem = testItem;
            }
        }

        return acceptedItem;

    }

    private static <T> RemoveOutcomeResult<ValueItem<T>> removeOutcomeItem(List<ValueItem<T>> items,
                                                                           T item,
                                                                           BiFunction<T, T, Boolean> equalityOverride) {

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double total = 0.0;

        List<ValueItem<T>> newItems = new ArrayList<>();

        for (ValueItem<T> valueItem : items) {

            if (!areEqual(item, valueItem.item, equalityOverride)) {

                newItems.add(valueItem);

                min = Math.min(min, valueItem.getValue());
                max = Math.max(max, valueItem.getValue());
                total += valueItem.getValue();
            }
        }

        if (newItems.size() < 1) {
            throw new IllegalStateException("No items remain");
        }

        DistributionStats stats = new DistributionStats(newItems.size(), min, max, total);
        return new RemoveOutcomeResult<>(newItems, stats);

    }

    private static <T> RemoveOutcomeResult<ValueGroup<T>> removeOutcomeGroup(
            List<ValueGroup<T>> groupedItems,
            T item,
            BiFunction<T, T, Boolean> equalityOverride) {

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double total = 0.0;

        List<ValueGroup<T>> newGroupedItems = new ArrayList<>();

        for (ValueGroup<T> group : groupedItems) {
            ValueGroup<T> newGroup = new ValueGroup<>(group.getValue());

            for (T groupItem : group.items) {
                if (!areEqual(item, groupItem, equalityOverride)) {
                    newGroup.items.add(groupItem);
                }
            }

            if (newGroup.items.size() > 0) {
                newGroupedItems.add(newGroup);

                min = Math.min(min, newGroup.getValue());
                max = Math.max(max, newGroup.getValue());
                total += newGroup.getValue();
            }
        }

        if (newGroupedItems.size() < 1) {
            throw new IllegalStateException("No items remain");
        }

        DistributionStats stats = new DistributionStats(newGroupedItems.size(), min, max, total);
        return new RemoveOutcomeResult<>(newGroupedItems, stats);

    }

    private static <T, S extends T> DiscreteDistribution<T> create(Iterable<S> items, Function<S, Double> getValue) {

        double min = Double.POSITIVE_INFINITY;
        double max = Double.NEGATIVE_INFINITY;
        double total = 0.0;

        double groupedMin = Double.POSITIVE_INFINITY;
        double groupedMax = Double.NEGATIVE_INFINITY;
        double groupedTotal = 0.0;

        List<ValueItem<T>> newItems = new ArrayList<>();
        List<ValueGroup<T>> newGroupedItems = new ArrayList<>();

        Map<Double, ValueGroup<T>> groupedItems = new HashMap<>();

        for (S item : items) {

            double value = getValue.apply(item);

            ValueItem<T> valueItem = new ValueItem<>(value, item);
            newItems.add(valueItem);

            min = Math.min(min, value);
            max = Math.max(max, value);
            total += value;

            ValueGroup<T> group = groupedItems.getOrDefault(value, null);

            if (group == null) {
                group = new ValueGroup<>(value);

                groupedItems.put(value, group);
                newGroupedItems.add(group);

                groupedMin = Math.min(groupedMin, value);
                groupedMax = Math.max(groupedMax, value);
                groupedTotal += value;
            }

            group.items.add(item);
        }

        if ((newItems.size() < 1) || (newGroupedItems.size() < 1)) {
            throw new IllegalArgumentException("No items were provided");
        }

        DistributionStats stats = new DistributionStats(newItems.size(), min, max, total);

        DistributionStats groupedStats = new DistributionStats(
                groupedItems.size(),
                groupedMin,
                groupedMax,
                groupedTotal);

        return new DiscreteDistribution<>(newItems, stats, newGroupedItems, groupedStats);

    }

    private static <T> boolean areEqual(T a, T b, BiFunction<T, T, Boolean> equalityOverride) {
        if (null != equalityOverride) {
            return equalityOverride.apply(a, b);
        }

        return a.equals(b);
    }

    private static class RemoveOutcomeResult<T> {

        final List<T> result;

        final DistributionStats stats;

        RemoveOutcomeResult(List<T> result, DistributionStats stats) {

            this.result = result;
            this.stats = stats;
        }

    }

    private static class ValueItem<T> implements DiscreteDistributionItem {

        private final double value;

        final T item;

        ValueItem(double value, T item) {
            this.value = value;
            this.item = item;
        }

        public double getValue() {
            return value;
        }

    }

    private static class ValueGroup<T> implements DiscreteDistributionItem {

        private final double value;

        final List<T> items;

        ValueGroup(double value) {
            this.value = value;
            this.items = new ArrayList<>();
        }

        public double getValue() {
            return value;
        }

    }

}
