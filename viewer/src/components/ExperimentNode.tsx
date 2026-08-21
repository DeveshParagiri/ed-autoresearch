import { memo } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export interface ExperimentNodeData extends Record<string, unknown> {
  title: string;
  question: string;
  selected: boolean;
  dimmed: boolean;
  onSelect: () => void;
}

export type ExperimentFlowNode = Node<ExperimentNodeData, "experiment">;

function ExperimentNodeComponent({ data }: NodeProps<ExperimentFlowNode>) {
  return (
    <>
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <button
        className={`experiment-node${data.selected ? " is-selected" : ""}${data.dimmed ? " is-dimmed" : ""}`}
        type="button"
        onClick={data.onSelect}
        aria-pressed={data.selected}
        aria-label={`${data.title}. ${data.question}`}
      >
        <span className="experiment-node__title">{data.title}</span>
        <span className="experiment-node__question">{data.question}</span>
      </button>
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </>
  );
}

export const ExperimentNode = memo(ExperimentNodeComponent);
