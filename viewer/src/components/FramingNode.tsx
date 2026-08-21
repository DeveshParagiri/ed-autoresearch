import { memo } from "react";
import type { Node, NodeProps } from "@xyflow/react";

export interface FramingNodeData extends Record<string, unknown> {
  title: string;
}

export type FramingFlowNode = Node<FramingNodeData, "framing">;

function FramingNodeComponent({ data }: NodeProps<FramingFlowNode>) {
  return (
    <div className="framing-node" aria-hidden="true">
      <span>{data.title}</span>
    </div>
  );
}

export const FramingNode = memo(FramingNodeComponent);
