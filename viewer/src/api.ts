import type { ExperimentDocumentData, ProjectSnapshot } from "./types";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
  return response.json() as Promise<T>;
}

export function loadProject(): Promise<ProjectSnapshot> {
  return getJson<ProjectSnapshot>("/api/autoresearch/graph");
}

export function loadExperiment(experimentId: string): Promise<ExperimentDocumentData> {
  return getJson<ExperimentDocumentData>(
    `/api/autoresearch/experiments/${encodeURIComponent(experimentId)}`,
  );
}

export async function revealArtifact(experimentId: string, reference: string): Promise<void> {
  const response = await fetch("/api/autoresearch/reveal", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ experimentId, reference }),
  });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
}

export function assetUrl(experimentId: string, reference: string, revision: number): string {
  const [pathPart] = reference.split(/[?#]/, 1);
  const encodedPath = pathPart
    .replace(/^\.\//, "")
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
  return `/api/autoresearch/assets/${encodeURIComponent(experimentId)}/${encodedPath}?v=${revision}`;
}

export function isExternalReference(reference: string): boolean {
  return /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(reference);
}

export function isImageReference(reference: string): boolean {
  if (/^data:image\//i.test(reference)) return true;
  const [pathPart] = reference.split(/[?#]/, 1);
  return /\.(?:avif|gif|jpe?g|png|svg|webp)$/i.test(pathPart);
}
