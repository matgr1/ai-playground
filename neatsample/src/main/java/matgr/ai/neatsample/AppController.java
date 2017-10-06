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
import matgr.ai.neatsample.minesweepers.MineField;
import matgr.ai.neatsample.minesweepers.MineSweeperSettings;
import matgr.ai.neatsample.neat.NeatMineSweeper;
import matgr.ai.neatsample.neat.NeatMineSweeperGeneticAlgorithm;
import matgr.ai.neatsample.neat.NeatMineSweeperPopulation;
import matgr.ai.neatsample.neat.NeatMineSweeperSpecies;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;

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

    // TODO: clean up synchronization stuff...
    private final Object statsLock;

    private CombinedMineSweeperStats writableCombinedStats;
    private Map<UUID, MineSweeperStatsItem> writableAllStats;

    private CombinedMineSweeperStats combinedStatsToSet;
    private Map<UUID, MineSweeperStatsItem> allStatsToSet;

    private Map<UUID, MineSweeperStatsItem> allStats;

    private boolean settingStats;

    public AppController() {

        statsLock = new Object();

        random = new MersenneTwister();
        settings = new MineSweeperSettings();

        geneticAlgorithm = new NeatMineSweeperGeneticAlgorithm(random, settings);

        evolutionContext = geneticAlgorithm.createEvolutionContext(settings.getEvolutionParameters(), settings.selectionStrategy);

        grapher = new NeuralNetGrapher();

        population = geneticAlgorithm.createRandomPopulation(
                evolutionContext,
                settings.populationCount,
                settings.getMineSweeperInputCount(),
                settings.mineSweeperOutputCount);

        setCombinedStats(new CombinedMineSweeperStats());
        setStats(new MineSweeperStatsItem());

        writableCombinedStats = new CombinedMineSweeperStats();

        writableAllStats = new HashMap<>();
        for (NeatMineSweeperSpecies species : population.species()) {
            for (NeatMineSweeper member : species.members()) {
                writableAllStats.put(member.genome().genomeId(), new MineSweeperStatsItem());
            }
        }

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

        long iteration = getCombinedStats().getIteration();

        MineSweeperStatsItem stats = allStats.get(sweeper.genome.genomeId());
        setStats(stats);

        // TODO: this is not safe... it draws the minefield which may be getting modified by the update thread... it
        //       happens to work OK now since the collection is not modified since mines are replace... won't work
        //       otherwise (and should be fixed regardless)
        MineSweeperScene.draw(mainCanvasSize, mainCanvasGraphics, sweeper, settings, iteration);

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

        if (!pauseEvolution) {

            if (writableCombinedStats.getGenerationIteration() > settings.iterationsPerGeneration) {

                bestSweeperGenomeId = null;

                NeatMineSweeperPopulation oldPopulation = population;
                population = geneticAlgorithm.evolve(evolutionContext, population);

                Iterator<NeatMineSweeper> oldMembers = new NestedIterator<>(oldPopulation.species(), Species::members);
                Iterator<NeatMineSweeper> newMembers = new NestedIterator<>(population.species(), Species::members);

                while (oldMembers.hasNext() && newMembers.hasNext()) {
                    NeatMineSweeper oldSweeper = oldMembers.next();
                    NeatMineSweeper newSweeper = newMembers.next();

                    newSweeper.initializeOnBirth(
                            oldSweeper.getPosition(),
                            oldSweeper.getDirection(),
                            writableCombinedStats.getIteration());
                }

                writableCombinedStats.setGeneration(writableCombinedStats.getGeneration() + 1);
                writableCombinedStats.setGenerationIteration(0L);
                writableCombinedStats.getCurrentGeneration().setExplosions(0);
                writableCombinedStats.getCurrentGeneration().setExplosions(0);

                writableAllStats = new HashMap<>();

                for (NeatMineSweeperSpecies species : population.species()) {
                    for (NeatMineSweeper member : species.members()) {
                        writableAllStats.put(member.genome().genomeId(), new MineSweeperStatsItem());
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
        long iteration = writableCombinedStats.getIteration();

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

                MineSweeperStatsItem curStatsItem = writableAllStats.get(mineSweeper.genome.genomeId());

                int explodeyMineIndex = mineField.isExplodeyHit(writableCombinedStats.getIteration(), settings, mineSweeper.getPosition());

                if (explodeyMineIndex >= 0) {

                    mineField.replaceExplodeyMine(random, explodeyMineIndex, writableCombinedStats.getIteration());
                    mineSweeper.onMineHit(writableCombinedStats.getIteration(), true);

                    currentExplosions++;
                    curStatsItem.setExplosions(mineSweeper.getExplosionCount());

                } else {

                    int mineIndex = mineField.isHit(writableCombinedStats.getIteration(), settings, mineSweeper.getPosition());

                    if (mineIndex >= 0) {

                        mineSweeper.onMineHit(writableCombinedStats.getIteration(), false);
                        //mineField.removeMine(mineIndex);
                        mineField.replaceMine(random, mineIndex, writableCombinedStats.getIteration());

                        currentMinesCleared++;
                        curStatsItem.setCleared(mineSweeper.getClearedCount());

                    } else {

                        mineSweeper.onMineNotHit(writableCombinedStats.getIteration());
                    }
                }

                curStatsItem.setFitness(mineSweeper.getFitness());

                // TODO: these don't make sense until prev species can be correlated with new species...
                //curStatsItem.setExplosionsPerIteration(curStatsItem.getTotals().getExplosions() / (double) curStatsItem.getIteration());
                //curStatsItem.setMinesClearedPerIteration(curStatsItem.getTotals().getCleared() / (double) curStatsItem.getIteration());
            }
        }

        // TODO: this is messy... clean it up
        writableCombinedStats.setIteration(writableCombinedStats.getIteration() + 1);
        writableCombinedStats.setGenerationIteration(writableCombinedStats.getGenerationIteration() + 1);

        writableCombinedStats.getTotals().setExplosions(writableCombinedStats.getTotals().getExplosions() + currentExplosions);
        writableCombinedStats.getCurrentGeneration().setExplosions(writableCombinedStats.getCurrentGeneration().getExplosions() + currentExplosions);

        writableCombinedStats.getTotals().setCleared(writableCombinedStats.getTotals().getCleared() + currentMinesCleared);
        writableCombinedStats.getCurrentGeneration().setCleared(writableCombinedStats.getCurrentGeneration().getCleared() + currentMinesCleared);

        writableCombinedStats.getTotals().setExplosionsPerIteration(writableCombinedStats.getTotals().getExplosions() / (double) writableCombinedStats.getIteration());
        writableCombinedStats.getCurrentGeneration().setExplosionsPerIteration(writableCombinedStats.getCurrentGeneration().getExplosions() / (double) writableCombinedStats.getGenerationIteration());

        writableCombinedStats.getTotals().setMinesClearedPerIteration(writableCombinedStats.getTotals().getCleared() / (double) writableCombinedStats.getIteration());
        writableCombinedStats.getCurrentGeneration().setMinesClearedPerIteration(writableCombinedStats.getCurrentGeneration().getCleared() / (double) writableCombinedStats.getGenerationIteration());

        writableCombinedStats.setSpeciesCount(fitnessSnapshot.getSpeciesCount());

        // NOTE: this is probably a little unclear... acquiring the lock later (within the runLater callback)
        //       should be fine since that runs on a different thread...
        // TODO: this is kinda messy, find a better way...
        synchronized (statsLock) {

            combinedStatsToSet = writableCombinedStats.clone();

            allStatsToSet = new HashMap<>();
            for (Map.Entry<UUID, MineSweeperStatsItem> entry : writableAllStats.entrySet()) {
                allStatsToSet.put(entry.getKey(), entry.getValue().clone());
            }

            if (!settingStats) {

                settingStats = true;

                Platform.runLater(() -> {

                    synchronized (statsLock) {

                        setCombinedStats(combinedStatsToSet);
                        allStats = allStatsToSet;

                        settingStats = false;
                    }
                });
            }
        }
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
