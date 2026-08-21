import { useEffect, useMemo } from "react";
import dagre from "@dagrejs/dagre";
import {
  Controls,
  ReactFlow,
  useNodesInitialized,
  useReactFlow,
  useStore,
  type Edge,
  type Node,
  type NodeTypes,
} from "@xyflow/react";

import type { ExperimentSummary, FramingSummary } from "../types";
import { ExperimentNode, type ExperimentFlowNode } from "./ExperimentNode";
import { FramingNode, type FramingFlowNode } from "./FramingNode";

const NODE_WIDTH = 280;
const MIN_NODE_HEIGHT = 112;
const FRAMING_PADDING = 28;
const FRAMING_LABEL_SPACE = 30;

const nodeTypes: NodeTypes = {
  experiment: ExperimentNode,
  framing: FramingNode,
};

function estimatedHeight(experiment: ExperimentSummary): number {
  const titleLines = Math.max(1, Math.ceil(experiment.title.length / 28));
  const questionLines = Math.max(2, Math.ceil(experiment.question.length / 41));
  return Math.max(MIN_NODE_HEIGHT, 28 + titleLines * 18 + questionLines * 17);
}

function activeAncestors(experiments: ExperimentSummary[], selectedId?: string): Set<string> {
  if (!selectedId) return new Set();
  const byId = new Map(experiments.map((experiment) => [experiment.id, experiment]));
  const active = new Set<string>();
  const visitParent = (id: string) => {
    if (active.has(id)) return;
    active.add(id);
    for (const parent of byId.get(id)?.parents ?? []) visitParent(parent);
  };
  visitParent(selectedId);
  return active;
}

function createFlowElements(
  experiments: ExperimentSummary[],
  framings: FramingSummary[],
  selectedId: string | undefined,
  onSelect: (id: string) => void,
): { nodes: Node[]; edges: Edge[] } {
  const ids = new Set(experiments.map((experiment) => experiment.id));
  const graph = new dagre.graphlib.Graph();
  graph.setGraph({ rankdir: "TB", nodesep: 46, ranksep: 72, marginx: 38, marginy: 38 });
  graph.setDefaultEdgeLabel(() => ({}));

  const heights = new Map<string, number>();
  for (const experiment of experiments) {
    const height = estimatedHeight(experiment);
    heights.set(experiment.id, height);
    graph.setNode(experiment.id, { width: NODE_WIDTH, height });
  }
  for (const experiment of experiments) {
    for (const parent of experiment.parents.filter((id) => ids.has(id))) {
      graph.setEdge(parent, experiment.id);
    }
  }
  dagre.layout(graph);

  const active = activeAncestors(experiments, selectedId);
  const hasSelection = Boolean(selectedId);
  const experimentNodes: ExperimentFlowNode[] = experiments.map((experiment) => {
    const position = graph.node(experiment.id) as { x: number; y: number };
    const height = heights.get(experiment.id) ?? MIN_NODE_HEIGHT;
    return {
      id: experiment.id,
      type: "experiment",
      position: { x: position.x - NODE_WIDTH / 2, y: position.y - height / 2 },
      style: { width: NODE_WIDTH, pointerEvents: "all" },
      data: {
        title: experiment.title,
        question: experiment.question,
        selected: experiment.id === selectedId,
        dimmed: hasSelection && !active.has(experiment.id),
        onSelect: () => onSelect(experiment.id),
      },
      draggable: false,
      selectable: false,
      focusable: false,
      zIndex: 2,
    };
  });

  const byId = new Map(experimentNodes.map((node) => [node.id, node]));
  const framingById = new Map(framings.map((framing) => [framing.id, framing]));
  const framingNodes: FramingFlowNode[] = [];
  for (const framingId of new Set(experiments.map((item) => item.framing).filter(Boolean))) {
    if (!framingId) continue;
    const members = experiments
      .filter((experiment) => experiment.framing === framingId)
      .map((experiment) => byId.get(experiment.id))
      .filter((node): node is ExperimentFlowNode => Boolean(node));
    if (members.length === 0) continue;
    const left = Math.min(...members.map((node) => node.position.x));
    const top = Math.min(...members.map((node) => node.position.y));
    const right = Math.max(...members.map((node) => node.position.x + NODE_WIDTH));
    const bottom = Math.max(
      ...members.map((node) => node.position.y + (heights.get(node.id) ?? MIN_NODE_HEIGHT)),
    );
    framingNodes.push({
      id: `framing:${framingId}`,
      type: "framing",
      position: { x: left - FRAMING_PADDING, y: top - FRAMING_PADDING - FRAMING_LABEL_SPACE },
      style: {
        width: right - left + FRAMING_PADDING * 2,
        height: bottom - top + FRAMING_PADDING * 2 + FRAMING_LABEL_SPACE,
      },
      data: { title: framingById.get(framingId)?.title ?? framingId },
      draggable: false,
      selectable: false,
      focusable: false,
      zIndex: 0,
    });
  }

  const edges: Edge[] = experiments.flatMap((experiment) =>
    experiment.parents
      .filter((parent) => ids.has(parent))
      .map((parent) => ({
        id: `${parent}:${experiment.id}`,
        source: parent,
        target: experiment.id,
        className:
          active.has(parent) && active.has(experiment.id) ? "research-edge is-active" : "research-edge",
        zIndex: 1,
      })),
  );

  return { nodes: [...framingNodes, ...experimentNodes], edges };
}

interface ResearchDagProps {
  experiments: ExperimentSummary[];
  framings: FramingSummary[];
  selectedId?: string;
  splitView: boolean;
  onSelect: (id: string) => void;
  onDismiss: () => void;
}

interface ViewportControllerProps {
  splitView: boolean;
}

function ViewportController({ splitView }: ViewportControllerProps) {
  const nodesInitialized = useNodesInitialized();
  const flowRoot = useStore((state) => state.domNode);
  const { fitView } = useReactFlow();

  useEffect(() => {
    if (!nodesInitialized || !flowRoot) return;
    let animationFrame = 0;
    const fitGraph = () => {
      cancelAnimationFrame(animationFrame);
      animationFrame = requestAnimationFrame(() => {
        void fitView({
          padding: 0.14,
          maxZoom: 1,
          duration: splitView && !document.body.classList.contains("is-resizing") ? 120 : 0,
        });
      });
    };
    const resizeObserver = new ResizeObserver(fitGraph);
    resizeObserver.observe(flowRoot);
    fitGraph();
    return () => {
      cancelAnimationFrame(animationFrame);
      resizeObserver.disconnect();
    };
  }, [fitView, flowRoot, nodesInitialized, splitView]);

  return null;
}

export function ResearchDag({
  experiments,
  framings,
  selectedId,
  splitView,
  onSelect,
  onDismiss,
}: ResearchDagProps) {
  const { nodes, edges } = useMemo(
    () => createFlowElements(experiments, framings, selectedId, onSelect),
    [experiments, framings, selectedId, onSelect],
  );

  return (
    <ReactFlow
      key={splitView ? "split" : "full"}
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.18, maxZoom: 1 }}
      minZoom={0.25}
      maxZoom={1.35}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      zoomOnDoubleClick={false}
      onPaneClick={splitView ? onDismiss : undefined}
      proOptions={{ hideAttribution: true }}
      aria-label="Experiment lineage"
    >
      <Controls
        position="bottom-left"
        orientation="horizontal"
        showFitView={false}
        showInteractive={false}
      />
      <ViewportController splitView={splitView} />
    </ReactFlow>
  );
}
