package matgr.ai.genetic;

import matgr.ai.genetic.selection.SelectionStrategy;

public class EvolutionContext {

    public final EvolutionParameters evolutionParameters;

    public final SelectionStrategy selectionStrategy;

    public EvolutionContext(EvolutionParameters evolutionParameters,
                            SelectionStrategy selectionStrategy) {
        this.evolutionParameters = evolutionParameters;
        this.selectionStrategy = selectionStrategy;
    }

}
