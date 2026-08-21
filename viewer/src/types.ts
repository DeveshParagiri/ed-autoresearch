export interface ExperimentSummary {
  id: string;
  title: string;
  kind: string;
  question: string;
  parents: string[];
  framing?: string;
}

export interface FramingSummary {
  id: string;
  title: string;
}

export interface ProjectSnapshot {
  experiments: ExperimentSummary[];
  framings: FramingSummary[];
  revision: number;
}

export interface ExperimentDocumentData {
  id: string;
  title: string;
  body: string;
}
