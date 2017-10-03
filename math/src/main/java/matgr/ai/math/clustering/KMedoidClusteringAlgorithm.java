package matgr.ai.math.clustering;

import com.google.common.collect.ObjectArrays;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

// TODO: allow different strategies for initializing the first set of medoids

public abstract class KMedoidClusteringAlgorithm<ItemT> {

    public List<Cluster<ItemT>> compute(List<ItemT> items, int clusterCount) {

        if (null == items) {
            throw new IllegalArgumentException("items not provided");
        }
        if (clusterCount <= 0) {
            throw new IllegalArgumentException("invalid clusterCount");
        }
        if (clusterCount > items.size()) {
            throw new IllegalArgumentException(
                    String.format("Cannot create %d clusters for %d items", clusterCount, items.size()));
        }

        double[][] distances = computeDistances(items);

        // initialize ClusterItems and find initial medoids

        // NOTE: this is k-medoids from here: https://www.researchgate.net/publication/220215167_A_simple_and_fast_algorithm_for_K-medoids_clustering (A simple and
        //		 fast algorithm for K-medoids clustering)

        List<ClusterItem> clusterItems = createFixedSizeClusterItemArrayList(items.size());
        List<InitialSortValue> sortValues = createFixedSizeInitialSortValueArrayList(items.size());

        // TODO: parallelize the outer loop
        // TODO: this might be computing the same stuff a bunch in the inner loop? if so: compute the repeated stuff
        //       once and store in an array

        for (int j = 0; j < items.size(); j++) {
            double v_j_sum = 0.0;

            for (int i = 0; i < items.size(); i++) {
                double d_il_sum = 0.0;

                for (int l = 0; l < items.size(); l++) {
                    double d_il = distances[i][l];
                    d_il_sum += d_il;
                }

                double d_ij = distances[i][j];
                double q = d_ij / d_il_sum;

                v_j_sum += q;
            }

            ItemT item_j = items.get(j);

            clusterItems.set(j, new ClusterItem(item_j, j));
            sortValues.set(j, new InitialSortValue(j, v_j_sum));
        }

        sortValues.sort(Comparator.comparingDouble(a -> a.value));

        List<ClusterItem> initialMedoids = createFixedSizeClusterItemArrayList(clusterCount);
        for (int i = 0; i < clusterCount; i++) {
            InitialSortValue sortValue = sortValues.get(i);
            ClusterItem initialMedoid = clusterItems.get(sortValue.index);

            initialMedoids.set(i, initialMedoid);
        }

        // create initial clusters
        AssignToMedoidsResult assignResult = assignToMedoids(clusterItems, distances, initialMedoids);
        double initialDistanceSum = assignResult.medoidDistanceSum;
        List<List<ClusterItem>> initialClusters = assignResult.value;

        return compute(distances, clusterItems, initialMedoids, initialClusters, initialDistanceSum);
    }

    public List<Cluster<ItemT>> refine(List<List<ItemT>> clusters) {

        List<ItemT> allItems = new ArrayList<>();
        List<ClusterItem> clusterItems = new ArrayList<>();
        List<List<ClusterItem>> initialClusters = new ArrayList<>();

        int originalIndex = 0;

        for (List<ItemT> cluster : clusters) {

            List<ClusterItem> initialCluster = new ArrayList<>();
            initialClusters.add(initialCluster);

            for (ItemT item : cluster) {
                allItems.add(item);

                ClusterItem clusterItem = new ClusterItem(item, originalIndex++);

                clusterItems.add(clusterItem);
                initialCluster.add(clusterItem);
            }
        }

        double[][] distances = computeDistances(allItems);
        List<ClusterItem> initialMedoids = updateMedoids(distances, initialClusters);

        AssignToMedoidsResult assignResult = assignToMedoids(clusterItems, distances, initialMedoids);
        double initialDistanceSum = assignResult.medoidDistanceSum;
        initialClusters = assignResult.value;

        return compute(distances, clusterItems, initialMedoids, initialClusters, initialDistanceSum);
    }

    protected abstract double computeDistance(ItemT a, ItemT b);

    private List<Cluster<ItemT>> compute(double[][] distances,
                                         List<ClusterItem> clusterItems,
                                         List<ClusterItem> initialMedoids,
                                         List<List<ClusterItem>> initialClusters,
                                         double initialDistanceSum) {
        List<ClusterItem> medoids = initialMedoids;

        double distanceSum = initialDistanceSum;
        List<List<ClusterItem>> clusters = initialClusters;

        while (true) {
            // update medoids
            List<ClusterItem> newMedoids = updateMedoids(distances, clusters);

            // assign object to medoids
            AssignToMedoidsResult assignResult = assignToMedoids(clusterItems, distances, newMedoids);
            double newDistanceSum = assignResult.medoidDistanceSum;
            List<List<ClusterItem>> newClusters = assignResult.value;

            // keep going if improving...
            if (newDistanceSum < distanceSum) {

                medoids = newMedoids;
                clusters = newClusters;
                distanceSum = newDistanceSum;

            } else {

                // things got worse - abort! (NOTE: this only finds LOCAL minima)
                break;

            }
        }

        List<Cluster<ItemT>> results = new ArrayList<>();

        for (int speciesIndex = 0; speciesIndex < medoids.size(); speciesIndex++) {

            List<ClusterItem> curCluster = clusters.get(speciesIndex);
            ClusterItem curMedoid = medoids.get(speciesIndex);

            List<ItemT> itemsForCluster = new ArrayList<>();
            for (ClusterItem clusterItem : curCluster) {
                itemsForCluster.add(clusterItem.item);
            }

            Cluster<ItemT> resultCluster = new Cluster<>(itemsForCluster, curMedoid.item);
            results.add(resultCluster);

        }

        return results;
    }

    private double[][] computeDistances(List<ItemT> items) {
        double[][] distances = new double[items.size()][items.size()];

        // NOTE: the matrix is symmetric around the diagonal

        // TODO: parallelize the outer loop
        for (int i = 0; i < items.size(); i++) {
            ItemT itemA = items.get(i);

            for (int j = 0; j <= i; j++) {
                ItemT itemB = items.get(j);

                double distance = computeDistance(itemA, itemB);
                distances[i][j] = distance;

                if (i != j) {
                    distances[j][i] = distance;
                }
            }
        }

        return distances;
    }

    private AssignToMedoidsResult assignToMedoids(List<ClusterItem> clusterItems,
                                                  double[][] allDistances,
                                                  List<ClusterItem> medoids) {
        // TODO: parallelize this?

        List<List<ClusterItem>> clusters = new ArrayList<>();
        for (int i = 0; i < medoids.size(); i++) {
            clusters.add(new ArrayList<>());
        }

        double minDistanceSum = 0.0;

        for (ClusterItem clusterItem : clusterItems) {
            int minDistanceClusterIndex = -1;
            double minDistance = Double.NaN;

            for (int i = 0; i < medoids.size(); i++) {
                ClusterItem medoid = medoids.get(i);

                if (clusterItemsReferenceEqual(clusterItem, medoid)) {
                    minDistanceClusterIndex = i;
                    minDistance = 0.0;
                    break;
                }

                double distance = allDistances[clusterItem.originalIndex][medoid.originalIndex];

                if (minDistanceClusterIndex < 0) {
                    minDistanceClusterIndex = i;
                    minDistance = distance;
                } else {
                    if (distance < minDistance) {
                        minDistanceClusterIndex = i;
                        minDistance = distance;
                    }
                }
            }

            List<ClusterItem> minDistanceCluster = clusters.get(minDistanceClusterIndex);

            minDistanceCluster.add(clusterItem);

            minDistanceSum += minDistance;
        }

        return new AssignToMedoidsResult(clusters, minDistanceSum);
    }

    private List<ClusterItem> updateMedoids(double[][] distances, List<List<ClusterItem>> clusters) {

        List<ClusterItem> medoids = createFixedSizeClusterItemArrayList(clusters.size());

        // TODO: parallelize the outer loop
        for (int i = 0; i < clusters.size(); i++) {
            List<ClusterItem> cluster = clusters.get(i);

            ClusterItem medoid = null;
            double minTotalDistanceToOthers = Double.NaN;

            for (ClusterItem clusterItem : cluster) {
                double curTotalDistanceToOthers = 0.0;

                for (ClusterItem otherClusterItem : cluster) {

                    if (!clusterItemsReferenceEqual(clusterItem, otherClusterItem)) {
                        double distance = distances[clusterItem.originalIndex][otherClusterItem.originalIndex];
                        curTotalDistanceToOthers += distance;
                    }

                }

                if (null == medoid) {

                    medoid = clusterItem;
                    minTotalDistanceToOthers = curTotalDistanceToOthers;

                } else {

                    if (curTotalDistanceToOthers < minTotalDistanceToOthers) {
                        medoid = clusterItem;
                        minTotalDistanceToOthers = curTotalDistanceToOthers;
                    }

                }
            }

            medoids.set(i, medoid);
        }

        // TODO: do this better
        // TODO: parallelize?
        for (int clusterIndex = 0; clusterIndex < clusters.size(); clusterIndex++) {

            if (medoids.get(clusterIndex) == null) {

                ClusterItem worstOutlier = null;
                double worstOutlierDistance = Double.NaN;

                Set<ItemT> usedOutliers = new HashSet<>();

                for (int otherClusterIndex = 0; otherClusterIndex < clusters.size(); otherClusterIndex++) {

                    if (clusterIndex != otherClusterIndex) {

                        List<ClusterItem> otherCluster = clusters.get(otherClusterIndex);

                        if (otherCluster.size() > 0) {

                            ClusterItem otherMedoid = medoids.get(otherClusterIndex);

                            if (null != otherMedoid) {

                                for (ClusterItem otherClusterItem : otherCluster) {

                                    // don't grab the medoid or any already taken outliers
                                    if ((!clusterItemsReferenceEqual(otherClusterItem, otherMedoid)) &&
                                            (!usedOutliers.contains(otherClusterItem.item))) {

                                        double distance = distances[otherMedoid.originalIndex][otherClusterItem.originalIndex];

                                        if (null == worstOutlier) {
                                            worstOutlier = otherClusterItem;
                                            worstOutlierDistance = distance;
                                        } else {
                                            if (distance > worstOutlierDistance) {
                                                worstOutlier = otherClusterItem;
                                                worstOutlierDistance = distance;
                                            }
                                        }

                                    }

                                }
                            }
                        }
                    }

                }

                if (null != worstOutlier) {
                    medoids.set(clusterIndex, worstOutlier);
                    usedOutliers.add(worstOutlier.item);
                }

            }
        }

        return medoids;
    }

    private boolean clusterItemsReferenceEqual(ClusterItem a, ClusterItem b) {
        return a.item == b.item;
    }

    private List<ClusterItem> createFixedSizeClusterItemArrayList(int size) {
        ClusterItem[] array = ObjectArrays.newArray(ClusterItem.class, size);
        return Arrays.asList(array);
    }

    private List<InitialSortValue> createFixedSizeInitialSortValueArrayList(int size) {
        InitialSortValue[] array = ObjectArrays.newArray(InitialSortValue.class, size);
        return Arrays.asList(array);
    }

    private class AssignToMedoidsResult {
        public final List<List<ClusterItem>> value;
        public final double medoidDistanceSum;

        private AssignToMedoidsResult(List<List<ClusterItem>> value, double medoidDistanceSum) {
            this.value = value;
            this.medoidDistanceSum = medoidDistanceSum;
        }
    }

    private class InitialSortValue {
        public int index;

        public double value;

        public InitialSortValue(int index, double value) {
            this.index = index;
            this.value = value;
        }
    }

    private class ClusterItem {
        public final ItemT item;

        public final int originalIndex;

        public ClusterItem(ItemT item, int originalIndex) {
            this.item = item;
            this.originalIndex = originalIndex;
        }
    }
}
