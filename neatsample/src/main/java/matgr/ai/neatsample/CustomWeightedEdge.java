package matgr.ai.neatsample;

import org.jgrapht.graph.DefaultWeightedEdge;

public class CustomWeightedEdge extends DefaultWeightedEdge {
    @Override
    public String toString() {
        return String.format("%.2f", getWeight());
    }
}
