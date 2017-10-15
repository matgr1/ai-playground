package matgr.ai.neatsample;

import com.jgraph.layout.JGraphFacade;
import com.jgraph.layout.hierarchical.JGraphHierarchicalLayout;
import javafx.embed.swing.SwingNode;
import matgr.ai.neat.NeatNeuralNet;
import matgr.ai.neuralnet.Connection;
import matgr.ai.neuralnet.cyclic.CyclicNeuralNet;
import matgr.ai.neuralnet.Neuron;
import org.jgraph.JGraph;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.GraphConstants;
import org.jgrapht.ext.JGraphModelAdapter;
import org.jgrapht.graph.DefaultDirectedWeightedGraph;

import javax.swing.*;
import java.awt.*;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.HashMap;
import java.util.Map;
import java.util.logging.Logger;

public class NeuralNetGrapher {

    private static Logger logger;

    static {
        logger = Logger.getLogger(NeuralNetGrapher.class.getName());
    }

    private JGraph graphControl;

    private JPanel graphControlContainer;

    private Dimension renderedSize;
    private double renderedZoom;
    private NeatNeuralNet renderedGraph;

    private NeatNeuralNet graphToRender;
    private boolean renderingGraph;

    public NeuralNetGrapher() {
        renderingGraph = false;

        graphControl = new JGraph();
        graphControl.setAutoResizeGraph(true);
        graphControl.setAutoscrolls(true);

        // TODO: maybe do a scrollpane in JavaFX-land?
        graphControlContainer = new JPanel();
        graphControlContainer.add(graphControl);
    }

    public void renderTo(SwingNode graphContainer,
                         Dimension size,
                         double zoom,
                         NeatNeuralNet neuralNet) {

        graphToRender = neuralNet;

        if (!renderingGraph) {

            renderingGraph = true;

            SwingUtilities.invokeLater(() -> {

                try {

                    renderToSwingNode(graphContainer, size, zoom);
                } catch (Exception e) {

                    StringWriter stackTraceWriter = new StringWriter();
                    e.printStackTrace(new PrintWriter(stackTraceWriter));

                    String stackTrace = stackTraceWriter.toString();

                    logger.warning(String.format("Failed to render graph\n%s", stackTrace));

                } finally {

                    renderingGraph = false;
                }
            });
        }
    }

    private void renderToSwingNode(SwingNode graphContainer, Dimension size, double zoom) {

        // TODO: this could use some styling...

        boolean sizeChanged = false;
        boolean zoomChanged = false;
        boolean graphChanged = false;

        if (renderedSize == null) {

            sizeChanged = true;

        } else {

            if ((size.width != renderedSize.width) || (size.height != renderedSize.height)) {

                sizeChanged = true;
            }
        }

        if (zoom != renderedZoom) {
            zoomChanged = true;
        }

        if (graphChanged(graphToRender, renderedGraph)) {

            DefaultDirectedWeightedGraph<NeuronVertex, CustomWeightedEdge> graph =
                    new DefaultDirectedWeightedGraph<>(CustomWeightedEdge.class);

            Map<Long, NeuronVertex> neuronVertexMap = new HashMap<>();

            for (Neuron n : graphToRender.neurons.values()) {
                NeuronVertex v = new NeuronVertex(n);
                neuronVertexMap.put(n.id, v);
                graph.addVertex(v);
            }

            for (Connection c : graphToRender.connections.values()) {

                NeuronVertex source = neuronVertexMap.get(c.sourceId);
                NeuronVertex target = neuronVertexMap.get(c.targetId);

                CustomWeightedEdge edge = graph.addEdge(source, target);
                graph.setEdgeWeight(edge, c.weight);
            }

            JGraphModelAdapter graphModel = new JGraphModelAdapter<>(graph);

            Map graphModelAttributes = graphModel.getAttributes();
            GraphConstants.setAutoSize(graphModelAttributes, true);

            for (NeuronVertex v : graph.vertexSet()) {
                DefaultGraphCell cell = graphModel.getVertexCell(v);
                Map attr = cell.getAttributes();

                // TODO: this didn't work...
                GraphConstants.setSize(attr, new Dimension(50, 50));

                Map<DefaultGraphCell, Map> cellAttr = new HashMap<>();
                cellAttr.put(cell, attr);
                graphModel.edit(cellAttr, null, null, null);
            }

            graphControl.setModel(graphModel);

            graphChanged = true;
        }

        if (graphContainer.getContent() != graphControlContainer) {
            graphContainer.setContent(graphControlContainer);
        }

        if (sizeChanged || zoomChanged || graphChanged) {

            // set container size
            graphControlContainer.setPreferredSize(size);
            graphControlContainer.setMaximumSize(size);
            graphControlContainer.setMinimumSize(new Dimension(0, 0));
            graphControlContainer.setSize(size);

            // layout the graph
            graphControl.setPreferredSize(size);
            graphControl.setMaximumSize(size);
            graphControl.setMinimumSize(size);
            graphControl.setSize(size);

            graphControl.setScale(zoom);

            JGraphHierarchicalLayout layout = new JGraphHierarchicalLayout();
            layout.setOrientation(SwingConstants.WEST);

            JGraphFacade facade = new JGraphFacade(graphControl);
            layout.run(facade);

            Map nested = facade.createNestedMap(false, false);
            graphControl.getGraphLayoutCache().edit(nested);

            graphControl.refresh();
        }

        // updated rendered info
        renderedGraph = graphToRender;
        renderedSize = new Dimension(size.width, size.height);
        renderedZoom = zoom;
    }

    private static boolean graphChanged(CyclicNeuralNet graphToRender, CyclicNeuralNet renderedGraph) {

        if (renderedGraph == null) {
            return true;
        }

        if (graphToRender != renderedGraph) {
            return true;
        }

        if (graphToRender.version() != renderedGraph.version()) {
            return true;
        }

        return false;
    }

    static class NeuronVertex {

        public final Neuron neuron;

        public NeuronVertex(Neuron neuron) {
            this.neuron = neuron;
        }

        @Override
        public String toString() {
            return ((Long) neuron.id).toString();
        }
    }

}