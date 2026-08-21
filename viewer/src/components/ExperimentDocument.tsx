import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent,
} from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  assetUrl,
  isExternalReference,
  isImageReference,
  revealArtifact,
} from "../api";
import type { ExperimentDocumentData } from "../types";

interface ExperimentDocumentProps {
  document: ExperimentDocumentData;
  revision: number;
}

interface ActiveFigure {
  src: string;
  alt: string;
}

function figureLabel(reference: string): string {
  const [pathPart] = reference.split(/[?#]/, 1);
  const filename = pathPart.split("/").at(-1) || "figure";
  let decoded = filename;
  try {
    decoded = decodeURIComponent(filename);
  } catch {
    // Keep the encoded filename when the Markdown reference is malformed.
  }
  return decoded.replace(/\.[^.]+$/, "").replace(/[-_]+/g, " ");
}

export function ExperimentDocument({ document, revision }: ExperimentDocumentProps) {
  const articleRef = useRef<HTMLElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const figureStageRef = useRef<HTMLDivElement>(null);
  const figureTriggerRef = useRef<HTMLButtonElement | null>(null);
  const backdropPointerRef = useRef(false);
  const figureTogglePointerRef = useRef<"image" | "blank" | undefined>(undefined);
  const [activeFigure, setActiveFigure] = useState<ActiveFigure>();
  const [figureMagnified, setFigureMagnified] = useState(false);

  const closeFigure = useCallback((restoreFocus = true) => {
    if (dialogRef.current?.open) dialogRef.current.close();
    setFigureMagnified(false);
    setActiveFigure(undefined);
    if (restoreFocus) {
      requestAnimationFrame(() => figureTriggerRef.current?.focus());
    }
  }, []);

  useEffect(() => {
    articleRef.current?.parentElement?.scrollTo({ top: 0 });
    closeFigure(false);
  }, [closeFigure, document.id]);

  useEffect(() => {
    if (activeFigure && dialogRef.current && !dialogRef.current.open) {
      dialogRef.current.showModal();
    }
  }, [activeFigure]);

  useEffect(() => {
    if (figureMagnified) return;
    const frame = requestAnimationFrame(() => {
      figureStageRef.current?.scrollTo({ top: 0, left: 0 });
    });
    return () => cancelAnimationFrame(frame);
  }, [activeFigure?.src, figureMagnified]);

  const openFigure = useCallback((trigger: HTMLButtonElement, src: string, alt: string) => {
    figureTriggerRef.current = trigger;
    setFigureMagnified(false);
    setActiveFigure({ src, alt });
  }, []);

  const toggleFigureMagnification = useCallback(() => {
    setFigureMagnified((magnified) => !magnified);
  }, []);

  const startBackdropClick = useCallback((event: PointerEvent<HTMLDialogElement>) => {
    backdropPointerRef.current = event.target === event.currentTarget;
    event.stopPropagation();
  }, []);

  const finishBackdropClick = useCallback(
    (event: PointerEvent<HTMLDialogElement>) => {
      const shouldClose = backdropPointerRef.current && event.target === event.currentTarget;
      backdropPointerRef.current = false;
      event.stopPropagation();
      if (shouldClose) closeFigure();
    },
    [closeFigure],
  );

  return (
    <article className="experiment-document" ref={articleRef}>
      <h1 className="experiment-document__title">{document.title}</h1>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        skipHtml
        components={{
          a({ href = "", children, ...props }) {
            if (!href) return <span>{children}</span>;
            if (isImageReference(href)) {
              const resolved = isExternalReference(href)
                ? href
                : assetUrl(document.id, href, revision);
              const label = figureLabel(href);
              return (
                <button
                  className="artifact-link"
                  type="button"
                  aria-label={`View figure: ${label}`}
                  title={`View ${label}`}
                  onClick={(event) => openFigure(event.currentTarget, resolved, label)}
                >
                  {children}
                </button>
              );
            }
            if (isExternalReference(href)) {
              const external = !href.startsWith("#");
              return (
                <a
                  href={href}
                  {...(external ? { target: "_blank", rel: "noreferrer" } : {})}
                  {...props}
                >
                  {children}
                </a>
              );
            }
            return (
              <button
                className="artifact-link"
                type="button"
                title="Show in file browser"
                onClick={() => void revealArtifact(document.id, href)}
              >
                {children}
              </button>
            );
          },
          img({ src = "", alt = "", ...props }) {
            if (!src) return null;
            const resolved = isExternalReference(src)
              ? src
              : assetUrl(document.id, src, revision);
            return (
              <button
                className="experiment-figure"
                type="button"
                aria-label={alt ? `View figure: ${alt}` : "View figure"}
                onClick={(event) => openFigure(event.currentTarget, resolved, alt)}
              >
                <img src={resolved} alt={alt} loading="lazy" {...props} />
              </button>
            );
          },
          table({ children, ...props }) {
            return (
              <div className="table-scroll">
                <table {...props}>{children}</table>
              </div>
            );
          },
        }}
      >
        {document.body}
      </ReactMarkdown>
      <dialog
        className="figure-viewer"
        ref={dialogRef}
        aria-label={activeFigure?.alt || "Figure viewer"}
        onCancel={(event) => {
          event.preventDefault();
          event.stopPropagation();
          closeFigure();
        }}
        onClick={(event) => event.stopPropagation()}
        onPointerDown={startBackdropClick}
        onPointerUp={finishBackdropClick}
      >
        <div className="figure-viewer__toolbar">
          <button
            className="figure-viewer__close"
            type="button"
            aria-label="Close figure viewer"
            title="Close figure viewer"
            onClick={() => closeFigure()}
          >
            <svg aria-hidden="true" viewBox="0 0 20 20">
              <path d="M4 4l12 12M16 4L4 16" />
            </svg>
          </button>
        </div>
        <div
          className={`figure-viewer__stage${figureMagnified ? " is-magnified" : ""}`}
          ref={figureStageRef}
        >
          {activeFigure ? (
            <div className="figure-viewer__canvas">
              <button
                className="figure-viewer__image-toggle"
                type="button"
                aria-label={figureMagnified ? "Fit figure to viewer" : "View figure at actual size"}
                title={figureMagnified ? "Fit figure to viewer" : "View figure at actual size"}
                onPointerDown={(event) => {
                  figureTogglePointerRef.current =
                    event.target === event.currentTarget ? "blank" : "image";
                }}
                onPointerCancel={() => {
                  figureTogglePointerRef.current = undefined;
                }}
                onPointerLeave={(event) => {
                  if (event.buttons) figureTogglePointerRef.current = undefined;
                }}
                onBlur={() => {
                  figureTogglePointerRef.current = undefined;
                }}
                onClick={() => {
                  const pointerTarget = figureTogglePointerRef.current;
                  figureTogglePointerRef.current = undefined;
                  if (pointerTarget !== "blank") toggleFigureMagnification();
                }}
              >
                <img
                  src={activeFigure.src}
                  alt={activeFigure.alt}
                  loading="eager"
                  draggable={false}
                />
              </button>
            </div>
          ) : null}
        </div>
      </dialog>
    </article>
  );
}
