package matgr.ai.genetic;

import matgr.ai.math.clustering.Cluster;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public class Species<SpeciesMemberT extends SpeciesMember<?>> {

    private final UUID speciesId;

    private final List<SpeciesMemberT> members;
    private final Map<UUID, SpeciesMemberT> memberMap;

    private final SpeciesMemberT representative;

    protected Species(Cluster<SpeciesMemberT> species) {

        this.speciesId = UUID.randomUUID();

        this.members = new ArrayList<>();
        this.memberMap = new HashMap<>();

        for (SpeciesMemberT member : species.items) {
            this.members.add(member);
            this.memberMap.put(member.genome().genomeId(), member);
        }

        this.representative = members.get(species.representativeIndex);
    }

    public UUID speciesId() {
        return speciesId;
    }

    public int memberCount() {
        return members.size();
    }

    public Iterable<SpeciesMemberT> members() {
        return members;
    }

    public boolean hasMember(UUID genomeId) {
        return memberMap.containsKey(genomeId);
    }

    public SpeciesMemberT getMember(int index) {
        return members.get(index);
    }

    public SpeciesMemberT representative() {
        return representative;
    }

    public SpeciesMemberT getMember(UUID genomeId) {
        return memberMap.get(genomeId);
    }
}
