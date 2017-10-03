package matgr.ai.neatsample;

import com.jgraph.layout.JGraphFacade;
import com.jgraph.layout.hierarchical.JGraphHierarchicalLayout;
import javafx.embed.swing.SwingNode;
import matgr.ai.neat.NeatNeuralNet;
import matgr.ai.neuralnet.cyclic.Connection;
import matgr.ai.neuralnet.cyclic.Neuron;
import org.jgraph.JGraph;
import org.jgraph.graph.DefaultGraphCell;
import org.jgraph.graph.GraphConstants;
import org.jgrapht.ext.JGraphModelAdapter;
import org.jgrapht.graph.ListenableDirectedWeightedGraph;

import javax.swing.*;
import java.awt.*;
import java.util.HashMap;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;

public class NeuralNetGrapher {

    private ListenableDirectedWeightedGraph<NeuronVertex, CustomWeightedEdge> graph;
    private JGraphModelAdapter graphModel;

    private JGraph graphControl;

    private JPanel graphControlContainer;

    private NeatNeuralNet graphToRender;
    private boolean renderingGraph;

    public NeuralNetGrapher(){
        renderingGraph = false;
    }

    public void renderTo(SwingNode swingNode,
                         Dimension size,
                         double zoom,
                         NeatNeuralNet neuralNet) {

        graphToRender = neuralNet;

        if (!renderingGraph) {

            renderingGraph = true;

            SwingUtilities.invokeLater(() -> {
                renderToSwingNode(swingNode, size, zoom, graphToRender);
                renderingGraph = false;
            });
        }
    }

    private void renderToSwingNode(SwingNode swingNode,
                                   Dimension size,
                                   double zoom,
                                   NeatNeuralNet neuralNet) {

        // TODO: this could use some styling...

        if (graph == null) {
            graph = new ListenableDirectedWeightedGraph<>(CustomWeightedEdge.class);

            Map<Long, NeuronVertex> neuronVertexMap = new HashMap<>();

            for (Neuron n : neuralNet.neurons.values()) {
                NeuronVertex v = new NeuronVertex(n);
                neuronVertexMap.put(n.id, v);
                graph.addVertex(v);
            }

            for (Connection c : neuralNet.connections.values()) {

                NeuronVertex source = neuronVertexMap.get(c.sourceId);
                NeuronVertex target = neuronVertexMap.get(c.targetId);

                CustomWeightedEdge edge = graph.addEdge(source, target);
                graph.setEdgeWeight(edge, c.weight);
            }
        } else {

            List<CustomWeightedEdge> currentEdges = new ArrayList<>();
            currentEdges.addAll(graph.edgeSet());

            List<NeuronVertex> currentVertices = new ArrayList<>();
            currentVertices.addAll(graph.vertexSet());

            graph.removeAllEdges(currentEdges);
            graph.removeAllVertices(currentVertices);

            // TODO: uncopy-paste
            Map<Long, NeuronVertex> neuronVertexMap = new HashMap<>();

            for (Neuron n : neuralNet.neurons.values()) {
                NeuronVertex v = new NeuronVertex(n);
                neuronVertexMap.put(n.id, v);
                graph.addVertex(v);
            }

            for (Connection c : neuralNet.connections.values()) {

                NeuronVertex source = neuronVertexMap.get(c.sourceId);
                NeuronVertex target = neuronVertexMap.get(c.targetId);

                CustomWeightedEdge edge = graph.addEdge(source, target);
                graph.setEdgeWeight(edge, c.weight);
            }
        }

        // create a visualization using JGraph, via an adapter
        if (graphModel == null) {
            graphModel = new JGraphModelAdapter<>(graph);

            Map graphModelAttributes = graphModel.getAttributes();
            GraphConstants.setAutoSize(graphModelAttributes, true);
        }

        if (graphControl == null) {

            graphControl = new JGraph();
            graphControl.setModel(graphModel);
            graphControl.setAutoResizeGraph(true);
            graphControl.setAutoscrolls(true);
        }

        if (graphControlContainer == null) {

            // TODO: maybe do a scrollpane in JavaFX-land?
            graphControlContainer = new JPanel();//jgraph);
            graphControlContainer.add(graphControl);

            swingNode.setContent(graphControlContainer);
        }

        // set container size
        graphControlContainer.setPreferredSize(size);
        graphControlContainer.setMaximumSize(size);
        graphControlContainer.setMinimumSize(size);
        graphControlContainer.setSize(size);

        // layout the graph
        graphControl.setPreferredSize(size);
        graphControl.setMaximumSize(size);
        graphControl.setMinimumSize(size);
        graphControl.setSize(size);

        graphControl.setScale(zoom);

        for (NeuronVertex v : graph.vertexSet()) {
            DefaultGraphCell cell = graphModel.getVertexCell(v);
            Map attr = cell.getAttributes();

            // TODO: this didn't work...
            GraphConstants.setSize(attr, new Dimension(50, 50));

            Map<DefaultGraphCell, Map> cellAttr = new HashMap<>();
            cellAttr.put(cell, attr);
            graphModel.edit(cellAttr, null, null, null);
        }

        JGraphHierarchicalLayout layout = new JGraphHierarchicalLayout();
        layout.setOrientation(SwingConstants.WEST);

        JGraphFacade facade = new JGraphFacade(graphControl);
        layout.run(facade);

        Map nested = facade.createNestedMap(false, false);
        graphControl.getGraphLayoutCache().edit(nested);

        graphControl.refresh();
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