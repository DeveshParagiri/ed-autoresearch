import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";

import { loadExperiment, loadProject } from "./api";
import { ExperimentDocument } from "./components/ExperimentDocument";
import { ResearchDag } from "./components/ResearchDag";
import type { ExperimentDocumentData, ProjectSnapshot } from "./types";

type LoadState = "loading" | "ready" | "error";
const PANEL_TRANSITION_MS = 180;

export default function App() {
  const [project, setProject] = useState<ProjectSnapshot>();
  const [selectedId, setSelectedId] = useState<string>();
  const [panelOpen, setPanelOpen] = useState(false);
  const [experimentDocument, setExperimentDocument] = useState<ExperimentDocumentData>();
  const [projectState, setProjectState] = useState<LoadState>("loading");
  const [documentState, setDocumentState] = useState<LoadState>("loading");
  const [graphWidth, setGraphWidth] = useState<number>();
  const [assetRevision, setAssetRevision] = useState(0);
  const selectedIdRef = useRef<string | undefined>(undefined);
  const documentCacheRef = useRef(new Map<string, ExperimentDocumentData>());
  const documentRequestRef = useRef(0);
  const shellRef = useRef<HTMLElement>(null);
  const closeTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  useEffect(() => {
    selectedIdRef.current = selectedId;
  }, [selectedId]);

  useEffect(
    () => () => {
      if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    },
    [],
  );

  const openExperiment = useCallback(
    (id: string) => {
      if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
      setPanelOpen(true);

      const requestId = ++documentRequestRef.current;
      const cachedDocument = documentCacheRef.current.get(id);
      if (cachedDocument) {
        selectedIdRef.current = id;
        setSelectedId(id);
        setExperimentDocument(cachedDocument);
        setDocumentState("ready");
        return;
      }

      const hasVisibleDocument = Boolean(experimentDocument);
      if (!hasVisibleDocument) {
        selectedIdRef.current = id;
        setSelectedId(id);
        setDocumentState("loading");
      }

      void loadExperiment(id)
        .then((nextDocument) => {
          documentCacheRef.current.set(id, nextDocument);
          if (requestId !== documentRequestRef.current) return;
          selectedIdRef.current = id;
          setSelectedId(id);
          setExperimentDocument(nextDocument);
          setDocumentState("ready");
        })
        .catch(() => {
          if (requestId === documentRequestRef.current && !hasVisibleDocument) {
            setDocumentState("error");
          }
        });
    },
    [experimentDocument],
  );

  const closeExperiment = useCallback(() => {
    documentRequestRef.current += 1;
    selectedIdRef.current = undefined;
    setPanelOpen(false);
    if (closeTimerRef.current) clearTimeout(closeTimerRef.current);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    closeTimerRef.current = setTimeout(
      () => {
        setSelectedId(undefined);
        setExperimentDocument(undefined);
        setDocumentState("ready");
      },
      reduceMotion ? 0 : PANEL_TRANSITION_MS,
    );
  }, []);

  useEffect(() => {
    if (!panelOpen) return;
    const closeOnEscape = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape" && !document.querySelector("dialog[open]")) closeExperiment();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [closeExperiment, panelOpen]);

  const refreshProject = useCallback(async () => {
    try {
      const nextProject = await loadProject();
      setProject(nextProject);
      setProjectState("ready");
      const currentSelection = selectedIdRef.current;
      const selectionStillExists = nextProject.experiments.some(
        (experiment) => experiment.id === currentSelection,
      );
      if (currentSelection && !selectionStillExists) {
        selectedIdRef.current = undefined;
        setPanelOpen(false);
        setSelectedId(undefined);
      }
    } catch {
      setProjectState("error");
    }
  }, []);

  useEffect(() => {
    void refreshProject();
  }, [refreshProject]);

  const clampGraphWidth = useCallback((clientX: number) => {
    const shell = shellRef.current;
    if (!shell) return;
    const bounds = shell.getBoundingClientRect();
    const minimumGraphWidth = Math.max(280, (bounds.width - 12) / 2);
    const minimumDocumentWidth = 360;
    const maximumGraphWidth = bounds.width - minimumDocumentWidth - 12;
    setGraphWidth(Math.max(minimumGraphWidth, Math.min(maximumGraphWidth, clientX - bounds.left)));
  }, []);

  const beginResize = useCallback(
    (event: ReactPointerEvent<HTMLButtonElement>) => {
      event.preventDefault();
      clampGraphWidth(event.clientX);
      document.body.classList.add("is-resizing");
      const move = (moveEvent: PointerEvent) => clampGraphWidth(moveEvent.clientX);
      const stop = () => {
        document.body.classList.remove("is-resizing");
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", stop);
        window.removeEventListener("pointercancel", stop);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", stop, { once: true });
      window.addEventListener("pointercancel", stop, { once: true });
    },
    [clampGraphWidth],
  );

  const resizeWithKeyboard = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const shell = shellRef.current;
      if (!shell) return;
      const currentWidth = graphWidth ?? (shell.getBoundingClientRect().width - 12) / 2;
      const direction = event.key === "ArrowLeft" ? -24 : 24;
      clampGraphWidth(shell.getBoundingClientRect().left + currentWidth + direction);
    },
    [clampGraphWidth, graphWidth],
  );

  const refreshSelectedDocument = useCallback(async () => {
    const experimentId = selectedIdRef.current;
    if (!experimentId) return;
    try {
      const nextDocument = await loadExperiment(experimentId);
      documentCacheRef.current.set(experimentId, nextDocument);
      if (selectedIdRef.current !== experimentId) return;
      setExperimentDocument(nextDocument);
      setDocumentState("ready");
    } catch {
      // Keep the last complete document visible while an authored file is being rewritten.
    }
  }, []);

  useEffect(() => {
    if (!import.meta.hot) return;
    const handleUpdate = (update?: { documentChanged?: boolean; assetChanged?: boolean }) => {
      void refreshProject();
      if (update?.documentChanged) void refreshSelectedDocument();
      if (update?.assetChanged) setAssetRevision((revision) => revision + 1);
    };
    import.meta.hot.on("autoresearch:update", handleUpdate);
    return () => import.meta.hot?.off("autoresearch:update", handleUpdate);
  }, [refreshProject, refreshSelectedDocument]);

  if (projectState === "error") {
    return <main className="viewer-message">The research viewer could not read this project.</main>;
  }

  if (projectState === "loading" || !project) {
    return <main className="viewer-message">Loading research…</main>;
  }

  if (project.experiments.length === 0) {
    return <main className="viewer-message">No experiments have been authored yet.</main>;
  }

  return (
    <main
      className={`viewer-shell${panelOpen ? " has-document" : ""}`}
      ref={shellRef}
      style={
        panelOpen && graphWidth
          ? ({ gridTemplateColumns: `${graphWidth}px 12px minmax(0, 1fr)` } satisfies CSSProperties)
          : undefined
      }
    >
      <section className="graph-pane" aria-label="Research graph">
        <div className="graph-intro">
          <p className="graph-intro__title">Research Graph</p>
        </div>
        <ResearchDag
          experiments={project.experiments}
          framings={project.framings}
          selectedId={selectedId}
          splitView={panelOpen}
          onSelect={openExperiment}
          onDismiss={closeExperiment}
        />
      </section>
      <button
        className="resize-handle"
        type="button"
        aria-label="Resize experiment document"
        aria-hidden={!panelOpen}
        title="Drag to resize"
        tabIndex={panelOpen ? 0 : -1}
        disabled={!panelOpen}
        onPointerDown={beginResize}
        onKeyDown={resizeWithKeyboard}
      />
      <section className="document-pane" aria-label="Experiment record" aria-hidden={!panelOpen}>
        <div className="document-toolbar">
          <button
            className="document-close"
            type="button"
            aria-label="Close experiment record"
            title="Close experiment record"
            tabIndex={panelOpen ? 0 : -1}
            onClick={closeExperiment}
          >
            <svg aria-hidden="true" viewBox="0 0 20 20">
              <path d="M4 4l12 12M16 4L4 16" />
            </svg>
          </button>
        </div>
        {selectedId && documentState === "error" && !experimentDocument ? (
          <div className="document-message">The experiment document could not be loaded.</div>
        ) : selectedId && !experimentDocument ? (
          <div className="document-message">Loading experiment…</div>
        ) : selectedId && experimentDocument ? (
          <ExperimentDocument document={experimentDocument} revision={assetRevision} />
        ) : null}
      </section>
    </main>
  );
}
