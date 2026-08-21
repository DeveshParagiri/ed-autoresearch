import { EventEmitter } from "node:events";
import { mkdirSync, realpathSync, symlinkSync, writeFileSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";

import type { ViteDevServer } from "vite";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  autoresearchSource,
  getExperimentDocument,
  readProjectSnapshot,
  revealCommandForPlatform,
  resolveReferencedAsset,
} from "./projectSource";

const temporaryRoots: string[] = [];

async function fixture(): Promise<{ root: string; experiment: string }> {
  const root = await mkdtemp(path.join(tmpdir(), "autoresearch-viewer-"));
  temporaryRoots.push(root);
  const experiment = path.join(root, "research", "experiments", "experiment.test");
  mkdirSync(path.join(experiment, "figures"), { recursive: true });
  mkdirSync(path.join(experiment, "private"), { recursive: true });
  writeFileSync(path.join(experiment, "figures", "result.png"), "image");
  writeFileSync(path.join(experiment, "metrics.json"), "{}");
  writeFileSync(path.join(experiment, "private", "trace.log"), "secret");
  mkdirSync(path.join(root, "research", "framings"), { recursive: true });
  writeFileSync(
    path.join(root, "research", "framings", "question.md"),
    "---\nid: framing.question\ntitle: Understand the fire mechanism\n---\n",
  );
  writeFileSync(
    path.join(experiment, "experiment.md"),
    `---
id: experiment.test
title: Test the viewer contract
kind: baseline
parents: []
---

# Question

Can the viewer render a complete authored question without deriving a second narrative?

# Evidence

![Result figure](figures/result.png)

[Metrics](metrics.json)
`,
  );
  return { root, experiment };
}

afterEach(async () => {
  await Promise.all(temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })));
});

describe("project source", () => {
  it("uses the root research.md when the project has one research document", async () => {
    const root = await mkdtemp(path.join(tmpdir(), "autoresearch-viewer-single-"));
    temporaryRoots.push(root);
    writeFileSync(
      path.join(root, "research.md"),
      `# Fire research

## Goal

Improve the model in a continuous loop.

## Working memory

Start tomorrow.
`,
    );

    expect(readProjectSnapshot(root).experiments).toEqual([
      expect.objectContaining({
        id: "research",
        title: "Fire research",
        kind: "research",
        question: "Improve the model in a continuous loop.",
        parents: [],
      }),
    ]);
    expect(getExperimentDocument(root, "research")?.body).toContain("## Working memory");
  });

  it("extracts human graph metadata and strips frontmatter from the document", async () => {
    const { root } = await fixture();
    const snapshot = readProjectSnapshot(root);
    expect(snapshot.experiments).toEqual([
      expect.objectContaining({
        id: "experiment.test",
        title: "Test the viewer contract",
        kind: "baseline",
        question: "Can the viewer render a complete authored question without deriving a second narrative?",
        parents: [],
      }),
    ]);
    expect(snapshot.framings).toEqual([
      { id: "framing.question", title: "Understand the fire mechanism" },
    ]);
    const document = getExperimentDocument(root, "experiment.test");
    expect(document?.body).toContain("# Question");
    expect(document?.body).not.toContain("id: experiment.test");
  });

  it("serves only files explicitly linked by experiment.md", async () => {
    const { root, experiment } = await fixture();
    expect(resolveReferencedAsset(root, "experiment.test", "figures/result.png")).toBe(
      realpathSync(path.join(experiment, "figures", "result.png")),
    );
    expect(resolveReferencedAsset(root, "experiment.test", "metrics.json")).toBe(
      realpathSync(path.join(experiment, "metrics.json")),
    );
    expect(resolveReferencedAsset(root, "experiment.test", "private/trace.log")).toBeUndefined();
  });

  it("rejects traversal, absolute paths, and unknown experiments", async () => {
    const { root, experiment } = await fixture();
    const outside = path.join(root, "outside.txt");
    writeFileSync(outside, "outside");
    symlinkSync(outside, path.join(experiment, "linked-outside.txt"));
    const experimentFile = path.join(experiment, "experiment.md");
    writeFileSync(
      experimentFile,
      `${writeFileSource()}\n[Linked outside](linked-outside.txt)\n`,
    );
    expect(resolveReferencedAsset(root, "experiment.test", "../outside.txt")).toBeUndefined();
    expect(resolveReferencedAsset(root, "experiment.test", "/etc/passwd")).toBeUndefined();
    expect(resolveReferencedAsset(root, "experiment.test", "linked-outside.txt")).toBeUndefined();
    expect(resolveReferencedAsset(root, "experiment.missing", "metrics.json")).toBeUndefined();
  });

  it("uses the native file browser on macOS, Windows, and Linux", () => {
    expect(revealCommandForPlatform("darwin", "/project/result.csv")).toEqual({
      command: "open",
      args: ["-R", "/project/result.csv"],
    });
    expect(revealCommandForPlatform("win32", "C:\\project\\result.csv")).toEqual({
      command: "explorer.exe",
      args: ["/select,C:\\project\\result.csv"],
    });
    expect(revealCommandForPlatform("linux", "/project/result.csv")).toEqual({
      command: "xdg-open",
      args: ["/project"],
    });
  });

  it("ignores raw run churn but refreshes authored sources and linked assets", async () => {
    vi.useFakeTimers();
    try {
      const { root, experiment } = await fixture();
      const watcher = Object.assign(new EventEmitter(), { add: vi.fn() });
      const send = vi.fn();
      const plugin = autoresearchSource(root);
      if (typeof plugin.configureServer !== "function") {
        throw new Error("Expected the project source to configure the Vite server.");
      }
      const configureServer = plugin.configureServer as unknown as (server: ViteDevServer) => void;
      configureServer({
        watcher,
        ws: { send },
        middlewares: { use: vi.fn() },
      } as unknown as ViteDevServer);

      watcher.emit(
        "change",
        path.join(experiment, "runs", "run.test", "work", "events.jsonl"),
      );
      watcher.emit("add", path.join(experiment, "runs", "run.test", "logs", "tool.log"));
      await vi.advanceTimersByTimeAsync(100);
      expect(send).not.toHaveBeenCalled();

      watcher.emit("change", path.join(experiment, "figures", "result.png"));
      await vi.advanceTimersByTimeAsync(100);
      expect(send).toHaveBeenLastCalledWith(
        expect.objectContaining({
          event: "autoresearch:update",
          data: expect.objectContaining({ assetChanged: true, documentChanged: false }),
        }),
      );

      send.mockClear();
      watcher.emit("change", path.join(experiment, "experiment.md"));
      await vi.advanceTimersByTimeAsync(100);
      expect(send).toHaveBeenCalledTimes(1);
      expect(send).toHaveBeenLastCalledWith(
        expect.objectContaining({
          event: "autoresearch:update",
          data: expect.objectContaining({ assetChanged: false, documentChanged: true }),
        }),
      );
    } finally {
      vi.useRealTimers();
    }
  });
});

function writeFileSource(): string {
  return `---
id: experiment.test
title: Test the viewer contract
kind: baseline
parents: []
---

# Question

Can the viewer render a complete authored question without deriving a second narrative?

# Evidence

![Result figure](figures/result.png)

[Metrics](metrics.json)`;
}
