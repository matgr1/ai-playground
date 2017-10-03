package matgr.ai.neatsample;

import javafx.animation.AnimationTimer;
import javafx.application.Platform;
import javafx.beans.property.IntegerProperty;
import javafx.beans.property.ReadOnlyObjectProperty;
import javafx.beans.property.ReadOnlyObjectWrapper;
import javafx.beans.property.ReadOnlyStringProperty;
import javafx.beans.property.ReadOnlyStringWrapper;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.embed.swing.SwingNode;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.CheckBox;
import javafx.scene.control.Slider;
import javafx.scene.layout.AnchorPane;
import matgr.ai.common.NestedIterator;
import matgr.ai.genetic.EvolutionContext;
import matgr.ai.genetic.PopulationFitnessSnapshot;
import matgr.ai.genetic.Species;
import matgr.ai.neat.NeatNeuralNet;
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;
import matgr.ai.neatsample.neat.NeatMineSweeper;
import matgr.ai.neatsample.neat.NeatMineSweeperGeneticAlgorithm;
import matgr.ai.neatsample.neat.NeatMineSweeperPopulation;
import matgr.ai.neatsample.neat.NeatMineSweeperSpecies;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;

public class AppController {

    @FXML
    private Slider graphZoomSlider;

    @FXML
    private AnchorPane graphContainerContainer;

    @FXML
    private SwingNode graphContainer;

    @FXML
    private CheckBox manualSpeciesCheckBox;

    @FXML
    private CheckBox pauseEvolutionCheckBox;

    @FXML
    private Slider manualSpeciesSlider;

    @FXML
    private Canvas mainCanvas;

    private ReadOnlyStringWrapper fps = new ReadOnlyStringWrapper("0 fps");

    public String getFps() {
        return fps.get();
    }

    private void setFps(String value) {
        fps.set(value);
    }

    public ReadOnlyStringProperty fpsProperty() {
        return fps.getReadOnlyProperty();
    }

    private ReadOnlyObjectWrapper<CombinedMineSweeperStats> combinedStats = new ReadOnlyObjectWrapper<>(new CombinedMineSweeperStats());

    public CombinedMineSweeperStats getCombinedStats() {
        return combinedStats.get();
    }

    private void setCombinedStats(CombinedMineSweeperStats value) {
        combinedStats.set(value);
    }

    public ReadOnlyObjectProperty<CombinedMineSweeperStats> combinedStatsProperty() {
        return combinedStats.getReadOnlyProperty();
    }

    private ReadOnlyObjectWrapper<MineSweeperStatsItem> stats = new ReadOnlyObjectWrapper<>(new MineSweeperStatsItem());

    public MineSweeperStatsItem getStats() {
        return stats.get();
    }

    private void setStats(MineSweeperStatsItem value) {
        stats.set(value);
    }

    public ReadOnlyObjectProperty<MineSweeperStatsItem> statsProperty() {
        return stats.getReadOnlyProperty();
    }

    private IntegerProperty selectedSpeciesIndex = new SimpleIntegerProperty();

    public Integer getSelectedSpeciesIndex() {
        return selectedSpeciesIndex.get();
    }

    public void setSelectedSpeciesIndex(Integer value) {
        selectedSpeciesIndex.set(value);
    }

    public IntegerProperty selectedSpeciesIndexProperty() {
        return selectedSpeciesIndex;
    }

    private final RandomGenerator random;

    private final MineSweeperSettings settings;

    private final NeatMineSweeperGeneticAlgorithm geneticAlgorithm;

    private final EvolutionContext evolutionContext;

    private final NeuralNetGrapher grapher;

    private NeatMineSweeperPopulation population;

    private final Thread updateLoopThread;
    private final ForkJoinPool updateExecutor;

    private final AnimationTimer animationTimer;

    private GraphicsContext mainCanvasGraphics;

    private boolean pauseEvolution;

    private Long lastTimestampNs;

    private UUID bestSweeperGenomeId;

    private Map<UUID, MineSweeperStatsItem> allStats;

    public AppController() {

        random = new MersenneTwister();
        settings = new MineSweeperSettings();

        geneticAlgorithm = new NeatMineSweeperGeneticAlgorithm(random, settings);

        evolutionContext = geneticAlgorithm.createEvolutionContext(settings.getEvolutionParameters(), settings.selectionStrategy);

        grapher = new NeuralNetGrapher();

        setCombinedStats(new CombinedMineSweeperStats());
        setStats(new MineSweeperStatsItem());

        population = geneticAlgorithm.createRandomPopulation(
                evolutionContext,
                settings.populationCount,
                settings.getMineSweeperInputCount(),
                settings.mineSweeperOutputCount,
                settings.activationResponse);


        allStats = new HashMap<>();

        for (NeatMineSweeperSpecies species : population.species()) {
            for (NeatMineSweeper member : species.members()) {
                allStats.put(member.genome().genomeId(), new MineSweeperStatsItem());
            }
        }

        AppController self = this;

        animationTimer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                self.handleAnimation(now);
            }
        };
        animationTimer.start();

        // NOTE: attempting to leave one core for UI updates while the heavy computation is happening
        int cores = Runtime.getRuntime().availableProcessors();
        int updateExecutorThreads = Math.max(cores - 1, 1);

        updateExecutor = new ForkJoinPool(updateExecutorThreads);

        updateLoopThread = Executors.defaultThreadFactory().newThread(this::updateLoop);
        updateLoopThread.setDaemon(true);
        updateLoopThread.start();
    }

    public void initialize() {
        mainCanvasGraphics = mainCanvas.getGraphicsContext2D();
    }

    private void handleAnimation(long now) {

        if (null != lastTimestampNs) {

            long diffNs = now - lastTimestampNs;

            double diffS = (double) diffNs / 1000000000.0;
            double fpsValue = 1.0 / diffS;

            setFps(String.format("%.2f FPS", fpsValue));
        }

        lastTimestampNs = now;

        Size mainCanvasSize = new Size(mainCanvas.getWidth(), mainCanvas.getHeight());

        NeatMineSweeper sweeper;

        if ((bestSweeperGenomeId == null) || manualSpeciesCheckBox.isSelected()) {

            // TODO: handle errors (eg index out of range...)
            int speciesIndex = (int) ((double) manualSpeciesSlider.getValue());
            NeatMineSweeperSpecies species = population.getSpecies(speciesIndex);

            sweeper = species.representative();

            for (NeatMineSweeper s : species.members()) {
                if (s.getFitness() > sweeper.getFitness()) {
                    sweeper = s;
                }
            }

            setSelectedSpeciesIndex(speciesIndex);

        } else {

            NeatMineSweeperSpecies bestSweeperSpecies = population.getGenomeSpecies(bestSweeperGenomeId);

            sweeper = bestSweeperSpecies.getMember(bestSweeperGenomeId);

            int speciesIndex = -1;

            for (NeatMineSweeperSpecies species : population.species()) {
                speciesIndex++;

                if (species.hasMember(sweeper.genome.genomeId())) {
                    break;
                }
            }

            manualSpeciesSlider.setValue(speciesIndex);
            setSelectedSpeciesIndex(speciesIndex);

        }

        setStats(allStats.get(sweeper.genome.genomeId()));

        MineSweeperScene.draw(mainCanvasSize, mainCanvasGraphics, sweeper, settings, getCombinedStats().getIteration());

        Dimension size = new Dimension(
                (int) Math.floor(graphContainerContainer.getWidth()),
                (int) Math.floor(graphContainerContainer.getHeight()));

        graphContainer.resize(size.width, size.height);
        grapher.renderTo(graphContainer, size, graphZoomSlider.getValue(), sweeper.genome.neuralNet);
    }


    private void updateLoop() {

        try {
            while (true) {
                try {
                    if (Thread.currentThread().isInterrupted()) {
                        break;
                    }

                    update();
                } catch (Exception e) {
                    // TODO: handle this
                    System.out.println("error in update call");
                }
            }
        } catch (Exception e) {
            // TODO: handle this
            System.out.println("error in update loop");
        } finally {
            System.out.println("Exiting update loop");
        }
    }

    private void update() {

        CombinedMineSweeperStats combinedStats = getCombinedStats();

        if (!pauseEvolution) {

            if (getCombinedStats().getGenerationIteration() > settings.iterationsPerGeneration) {

                bestSweeperGenomeId = null;

                NeatMineSweeperPopulation oldPopulation = population;
                population = geneticAlgorithm.evolve(evolutionContext, population);

                Iterator<NeatMineSweeper> oldMembers = new NestedIterator<>(oldPopulation.species(), Species::members);
                Iterator<NeatMineSweeper> newMembers = new NestedIterator<>(population.species(), Species::members);

                while (oldMembers.hasNext() && newMembers.hasNext()) {
                    NeatMineSweeper oldSweeper = oldMembers.next();
                    NeatMineSweeper newSweeper = newMembers.next();

                    newSweeper.initializeOnBirth(oldSweeper.getPosition(), oldSweeper.getDirection(), combinedStats.getIteration());
                }

                Platform.runLater(() ->
                {
                    combinedStats.setGeneration(combinedStats.getGeneration() + 1);
                    combinedStats.setGenerationIteration(0L);
                    combinedStats.getCurrentGeneration().setExplosions(0);
                    combinedStats.getCurrentGeneration().setExplosions(0);
                });

                allStats = new HashMap<>();

                for (NeatMineSweeperSpecies species : population.species()) {
                    for (NeatMineSweeper member : species.members()) {
                        allStats.put(member.genome().genomeId(), new MineSweeperStatsItem());
                    }
                }
            }
        }

        // TODO: this doesn't work, so just always getting best for now... need to get the best PREVIOUS species and
        //       correlated it with the new set somehow...
        //if (null == bestSweeperGenomeId) {
        bestSweeperGenomeId = getBestMineSweeperGenomeId(population, false);
        //}

        // update population
        long iteration = combinedStats.getIteration();

        List<CompletableFuture> futures = new ArrayList<>();

        for (NeatMineSweeperSpecies species : population.species()) {

            for (NeatMineSweeper mineSweeper : species.members()) {

                CompletableFuture future = CompletableFuture.runAsync(
                        () -> {
                            checkInUpdateExecutor();
                            mineSweeper.update(iteration);
                        },
                        updateExecutor);

                futures.add(future);
            }
        }

        CompletableFuture[] futureArray = futures.toArray(new CompletableFuture[futures.size()]);
        CompletableFuture.allOf(futureArray).join();

        // NOTE: this also seems to work for paralellizing the loops, and is probably a bit better on memory usage
        //       since it uses iterators instead of creating a list of the size of the whole population... memory
        //       usage shouldn't really be an issue here though
//        StreamSupport.stream(population.species().spliterator(), true).forEach(
//                species -> {
//                    StreamSupport.stream(species.members().spliterator(), true).forEach(
//                            mineSweeper -> {
//                                checkInUpdateExecutor();
//                                mineSweeper.update(iteration);
//                            }
//                    );
//                }
//        );

        PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);

        int currentExplosions = 0;
        int currentMinesCleared = 0;

        for (int i = 0; i < population.speciesCount(); i++) {

            NeatMineSweeperSpecies species = population.getSpecies(i);

            for (NeatMineSweeper mineSweeper : species.members()) {

                // TODO: this is a little weird (that the mine sweeper contains it's minefield...)
                MineField mineField = mineSweeper.getMineField();

                MineSweeperStatsItem curItem = allStats.get(mineSweeper.genome.genomeId());

                int explodeyMineIndex = mineField.isExplodeyHit(combinedStats.getIteration(), settings, mineSweeper.getPosition());

                if (explodeyMineIndex >= 0) {

                    mineField.replaceExplodeyMine(random, explodeyMineIndex, combinedStats.getIteration());
                    mineSweeper.onMineHit(combinedStats.getIteration(), true);

                    currentExplosions++;
                    Platform.runLater(() -> curItem.setExplosions(curItem.getExplosions() + 1));

                } else {

                    int mineIndex = mineField.isHit(combinedStats.getIteration(), settings, mineSweeper.getPosition());

                    if (mineIndex >= 0) {

                        mineSweeper.onMineHit(combinedStats.getIteration(), false);
                        mineField.replaceMine(random, mineIndex, combinedStats.getIteration());

                        currentMinesCleared++;
                        Platform.runLater(() -> curItem.setCleared(curItem.getCleared() + 1));

                    } else {

                        mineSweeper.onMineNotHit(combinedStats.getIteration());
                    }
                }

                Platform.runLater(() -> curItem.setFitness(mineSweeper.getFitness()));

                // TODO: these don't make sense until prev species can be correlated with new species...
                //curItem.setExplosionsPerIteration(curItem.getTotals().getExplosions() / (double) curItem.getIteration());
                //curItem.setMinesClearedPerIteration(curItem.getTotals().getCleared() / (double) curItem.getIteration());
            }
        }

        // TODO: this is messy... clean it up
        int[] currentExplosionsArray = new int[1];
        currentExplosionsArray[0] = currentExplosions;
        int[] currentMinesClearedArray = new int[1];
        currentMinesClearedArray[0] = currentMinesCleared;
        Platform.runLater(() ->
        {
            combinedStats.setIteration(combinedStats.getIteration() + 1);
            combinedStats.setGenerationIteration(combinedStats.getGenerationIteration() + 1);

            combinedStats.getTotals().setExplosions(combinedStats.getTotals().getExplosions() + currentExplosionsArray[0]);
            combinedStats.getCurrentGeneration().setExplosions(combinedStats.getCurrentGeneration().getExplosions() + currentExplosionsArray[0]);

            combinedStats.getTotals().setCleared(combinedStats.getTotals().getCleared() + currentMinesClearedArray[0]);
            combinedStats.getCurrentGeneration().setCleared(combinedStats.getCurrentGeneration().getCleared() + currentMinesClearedArray[0]);

            combinedStats.getTotals().setExplosionsPerIteration(combinedStats.getTotals().getExplosions() / (double) combinedStats.getIteration());
            combinedStats.getCurrentGeneration().setExplosionsPerIteration(combinedStats.getCurrentGeneration().getExplosions() / (double) combinedStats.getGenerationIteration());

            combinedStats.getTotals().setMinesClearedPerIteration(combinedStats.getTotals().getCleared() / (double) combinedStats.getIteration());
            combinedStats.getCurrentGeneration().setMinesClearedPerIteration(combinedStats.getCurrentGeneration().getCleared() / (double) combinedStats.getGenerationIteration());

            combinedStats.setSpeciesCount(fitnessSnapshot.getSpeciesCount());
        });
    }

    private void checkInUpdateExecutor() {
        if (ForkJoinTask.getPool() != updateExecutor) {
            throw new IllegalStateException("Not running in correct pool");
        }
    }

    private static UUID getBestMineSweeperGenomeId(NeatMineSweeperPopulation population, boolean useSpeciesRepresendative) {

        if (useSpeciesRepresendative) {
            NeatMineSweeperSpecies bestSpecies = getBestMineSweeperSpecies(population);

            if (bestSpecies == null) {
                return null;
            }

            return bestSpecies.representative().genome().genomeId();
        }

        PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);

        UUID bestId = null;
        double bestFitness = Double.NEGATIVE_INFINITY;

        for (NeatMineSweeperSpecies species : population.species()) {

            for (NeatMineSweeper sweeper : species.members()) {

                double fitness = fitnessSnapshot.getFitness(sweeper.genome);

                if (bestId == null) {

                    bestId = sweeper.genome().genomeId();
                    bestFitness = fitness;

                } else {

                    if (fitness > bestFitness) {

                        bestId = sweeper.genome().genomeId();
                        bestFitness = fitness;

                    }
                }
            }
        }

        return bestId;
    }

    private static NeatMineSweeperSpecies getBestMineSweeperSpecies(NeatMineSweeperPopulation population) {

        PopulationFitnessSnapshot fitnessSnapshot = PopulationFitnessSnapshot.create(population);

        NeatMineSweeperSpecies best = null;
        double bestFitness = Double.NEGATIVE_INFINITY;

        for (NeatMineSweeperSpecies species : population.species()) {

            double fitness = fitnessSnapshot.getFitness(species.representative().genome);

            if (best == null) {

                best = species;
                bestFitness = fitness;

            } else {

                if (fitness > bestFitness) {

                    best = species;
                    bestFitness = fitness;

                }
            }
        }

        return best;
    }

    @FXML
    private void onPauseEvolutionChecked() {
        this.pauseEvolution = pauseEvolutionCheckBox.isSelected();
    }
}
