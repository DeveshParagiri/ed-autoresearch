import {
  createReadStream,
  existsSync,
  readdirSync,
  readFileSync,
  realpathSync,
  statSync,
} from "node:fs";
import { spawn } from "node:child_process";
import path from "node:path";
import type { IncomingMessage, ServerResponse } from "node:http";

import matter from "gray-matter";
import { unified } from "unified";
import remarkParse from "remark-parse";
import { visit } from "unist-util-visit";
import type { Plugin, ViteDevServer } from "vite";

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

interface ExperimentRecord extends ExperimentSummary {
  directory: string;
  body: string;
  referencedAssets: Set<string>;
}

interface SourceIndex {
  experiments: Map<string, ExperimentRecord>;
  framings: FramingSummary[];
  revision: number;
}

interface SourceChange {
  refresh: boolean;
  documentChanged: boolean;
  assetChanged: boolean;
}

interface AstNode {
  type: string;
  depth?: number;
  value?: string;
  url?: string;
  children?: AstNode[];
}

const GRAPH_ENDPOINT = "/api/autoresearch/graph";
const DOCUMENT_PREFIX = "/api/autoresearch/experiments/";
const ASSET_PREFIX = "/api/autoresearch/assets/";
const REVEAL_ENDPOINT = "/api/autoresearch/reveal";

function childDirectories(directory: string): string[] {
  if (!existsSync(directory)) return [];
  return readdirSync(directory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(directory, entry.name))
    .sort();
}

function nodeText(node: AstNode): string {
  if (typeof node.value === "string") return node.value;
  return (node.children ?? []).map(nodeText).join("");
}

function extractQuestion(root: AstNode): string {
  const children = root.children ?? [];
  const questionIndex = children.findIndex(
    (node) =>
      node.type === "heading" &&
      node.depth === 1 &&
      nodeText(node).trim().toLowerCase() === "question",
  );
  if (questionIndex < 0) return "Question not yet authored.";

  const section: string[] = [];
  for (const node of children.slice(questionIndex + 1)) {
    if (node.type === "heading" && node.depth === 1) break;
    const text = nodeText(node).trim();
    if (text) section.push(text);
  }
  return section.join(" ") || "Question not yet authored.";
}

function extractSection(root: AstNode, title: string): string {
  const children = root.children ?? [];
  const sectionIndex = children.findIndex(
    (node) =>
      node.type === "heading" &&
      typeof node.depth === "number" &&
      nodeText(node).trim().toLowerCase() === title.toLowerCase(),
  );
  if (sectionIndex < 0) return "";
  const depth = children[sectionIndex].depth ?? 1;
  const section: string[] = [];
  for (const node of children.slice(sectionIndex + 1)) {
    if (node.type === "heading" && (node.depth ?? 1) <= depth) break;
    const text = nodeText(node).trim();
    if (text) section.push(text);
  }
  return section.join(" ");
}

function isRemoteReference(reference: string): boolean {
  return /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(reference);
}

function normalizeReference(reference: string): string | undefined {
  if (!reference || isRemoteReference(reference) || reference.includes("\0")) return undefined;
  const withoutSuffix = reference.split(/[?#]/, 1)[0];
  let decoded: string;
  try {
    decoded = decodeURIComponent(withoutSuffix);
  } catch {
    return undefined;
  }
  if (!decoded || path.isAbsolute(decoded)) return undefined;
  const normalized = path.posix.normalize(decoded.replaceAll("\\", "/"));
  if (normalized === ".." || normalized.startsWith("../")) return undefined;
  return normalized.replace(/^\.\//, "");
}

function collectReferencedAssets(root: AstNode): Set<string> {
  const assets = new Set<string>();
  visit(root as never, ["link", "image"], (node: { url?: string }) => {
    if (typeof node.url !== "string") return;
    const normalized = normalizeReference(node.url);
    if (normalized) assets.add(normalized);
  });
  return assets;
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && Boolean(item.trim()))
    : [];
}

function readExperiment(directory: string): ExperimentRecord | undefined {
  const filename = path.join(directory, "experiment.md");
  if (!existsSync(filename)) return undefined;
  try {
    const source = readFileSync(filename, "utf8");
    const parsed = matter(source);
    const tree = unified().use(remarkParse).parse(parsed.content) as AstNode;
    const fallbackId = path.basename(directory);
    return {
      id: stringValue(parsed.data.id, fallbackId),
      title: stringValue(parsed.data.title, "Untitled experiment"),
      kind: stringValue(parsed.data.kind, "experiment"),
      question: extractQuestion(tree),
      parents: stringArray(parsed.data.parents),
      ...(typeof parsed.data.framing === "string" && parsed.data.framing.trim()
        ? { framing: parsed.data.framing.trim() }
        : {}),
      directory,
      body: parsed.content.trim(),
      referencedAssets: collectReferencedAssets(tree),
    };
  } catch {
    return undefined;
  }
}

function readResearchDocument(projectRoot: string): ExperimentRecord | undefined {
  const filename = path.join(projectRoot, "research.md");
  if (!existsSync(filename)) return undefined;
  try {
    const source = readFileSync(filename, "utf8");
    const parsed = matter(source);
    const tree = unified().use(remarkParse).parse(parsed.content) as AstNode;
    const title =
      (tree.children ?? []).find((node) => node.type === "heading" && node.depth === 1);
    return {
      id: "research",
      title: title ? nodeText(title).trim() : "Research",
      kind: "research",
      question: extractSection(tree, "Goal") || "Goal not yet authored.",
      parents: [],
      directory: projectRoot,
      body: parsed.content.trim(),
      referencedAssets: collectReferencedAssets(tree),
    };
  } catch {
    return undefined;
  }
}

function readFramingFile(filename: string, fallbackId: string): FramingSummary | undefined {
  try {
    const parsed = matter(readFileSync(filename, "utf8"));
    const bodyTitle = parsed.content.match(/^#\s+(.+)$/m)?.[1]?.trim();
    return {
      id: stringValue(parsed.data.id, fallbackId),
      title: stringValue(parsed.data.title, bodyTitle ?? "Research framing"),
    };
  } catch {
    return undefined;
  }
}

function readFramings(root: string): FramingSummary[] {
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((entry) => {
      if (entry.isFile() && entry.name.endsWith(".md")) {
        return readFramingFile(path.join(root, entry.name), path.basename(entry.name, ".md"));
      }
      if (!entry.isDirectory()) return undefined;
      const directory = path.join(root, entry.name);
      const markdownFiles = readdirSync(directory, { withFileTypes: true })
        .filter((item) => item.isFile() && item.name.endsWith(".md"))
        .map((item) => path.join(directory, item.name))
        .sort();
      const filename =
        markdownFiles.find((item) => path.basename(item) === "framing.md") ?? markdownFiles[0];
      return filename ? readFramingFile(filename, entry.name) : undefined;
    })
    .filter((framing): framing is FramingSummary => Boolean(framing));
}

function scanProject(projectRoot: string): SourceIndex {
  const experimentRoot = path.join(projectRoot, "research", "experiments");
  const framingRoot = path.join(projectRoot, "research", "framings");
  const experiments = new Map<string, ExperimentRecord>();
  for (const directory of childDirectories(experimentRoot)) {
    const experiment = readExperiment(directory);
    if (experiment && !experiments.has(experiment.id)) experiments.set(experiment.id, experiment);
  }
  const research = readResearchDocument(projectRoot);
  if (research && !experiments.has(research.id)) experiments.set(research.id, research);
  const framings = readFramings(framingRoot);
  return { experiments, framings, revision: Date.now() };
}

function sourceChangeForPath(
  index: SourceIndex,
  researchFile: string,
  experimentRoot: string,
  framingRoot: string,
  changedPath: string,
): SourceChange {
  const absolute = path.resolve(changedPath);
  if (absolute === researchFile) {
    return { refresh: true, documentChanged: true, assetChanged: false };
  }
  const experimentRelative = path.relative(experimentRoot, absolute);
  const insideExperiments =
    experimentRelative !== "" &&
    experimentRelative !== ".." &&
    !experimentRelative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(experimentRelative);

  if (insideExperiments) {
    const parts = experimentRelative.split(path.sep);
    if (parts.length === 2 && parts[1] === "experiment.md") {
      return { refresh: true, documentChanged: true, assetChanged: false };
    }

    const referencedAsset = [...index.experiments.values()].some((experiment) =>
      [...experiment.referencedAssets].some(
        (reference) => path.resolve(experiment.directory, reference) === absolute,
      ),
    );
    if (referencedAsset) {
      return { refresh: true, documentChanged: false, assetChanged: true };
    }
  }

  const framingRelative = path.relative(framingRoot, absolute);
  const insideFramings =
    framingRelative !== "" &&
    framingRelative !== ".." &&
    !framingRelative.startsWith(`..${path.sep}`) &&
    !path.isAbsolute(framingRelative);
  if (insideFramings && path.extname(absolute).toLowerCase() === ".md") {
    return { refresh: true, documentChanged: false, assetChanged: false };
  }

  return { refresh: false, documentChanged: false, assetChanged: false };
}

export function readProjectSnapshot(projectRoot: string): ProjectSnapshot {
  const index = scanProject(projectRoot);
  return snapshotFromIndex(index);
}

function snapshotFromIndex(index: SourceIndex): ProjectSnapshot {
  return {
    experiments: [...index.experiments.values()]
      .map(({ directory: _directory, body: _body, referencedAssets: _assets, ...summary }) => summary)
      .sort((a, b) => a.title.localeCompare(b.title)),
    framings: [...index.framings].sort((a, b) => a.title.localeCompare(b.title)),
    revision: index.revision,
  };
}

export function getExperimentDocument(
  projectRoot: string,
  experimentId: string,
): { id: string; title: string; body: string } | undefined {
  const record = scanProject(projectRoot).experiments.get(experimentId);
  return record ? { id: record.id, title: record.title, body: record.body } : undefined;
}

export function resolveReferencedAsset(
  projectRoot: string,
  experimentId: string,
  requestedPath: string,
): string | undefined {
  const record = scanProject(projectRoot).experiments.get(experimentId);
  const normalized = normalizeReference(requestedPath);
  if (!record || !normalized || !record.referencedAssets.has(normalized)) return undefined;
  const absolute = path.resolve(record.directory, normalized);
  const relative = path.relative(record.directory, absolute);
  if (!relative || relative.startsWith("..") || path.isAbsolute(relative)) return undefined;
  if (!existsSync(absolute) || !statSync(absolute).isFile()) return undefined;
  const realDirectory = realpathSync(record.directory);
  const realAsset = realpathSync(absolute);
  const realRelative = path.relative(realDirectory, realAsset);
  if (!realRelative || realRelative.startsWith("..") || path.isAbsolute(realRelative)) return undefined;
  return realAsset;
}

export function revealCommandForPlatform(
  platform: NodeJS.Platform,
  filename: string,
): { command: string; args: string[] } {
  if (platform === "darwin") return { command: "open", args: ["-R", filename] };
  if (platform === "win32") return { command: "explorer.exe", args: [`/select,${filename}`] };
  return { command: "xdg-open", args: [path.dirname(filename)] };
}

function sendJson(response: ServerResponse, status: number, payload: unknown): void {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Cache-Control", "no-store");
  response.end(JSON.stringify(payload));
}

function mimeType(filename: string): string {
  switch (path.extname(filename).toLowerCase()) {
    case ".png": return "image/png";
    case ".jpg":
    case ".jpeg": return "image/jpeg";
    case ".gif": return "image/gif";
    case ".webp": return "image/webp";
    case ".svg": return "image/svg+xml";
    case ".json": return "application/json; charset=utf-8";
    case ".csv": return "text/csv; charset=utf-8";
    case ".txt": return "text/plain; charset=utf-8";
    case ".pdf": return "application/pdf";
    case ".nc": return "application/x-netcdf";
    default: return "application/octet-stream";
  }
}

function decodeSegment(value: string): string | undefined {
  try {
    return decodeURIComponent(value);
  } catch {
    return undefined;
  }
}

function handleAssetRequest(
  projectRoot: string,
  pathname: string,
  response: ServerResponse,
): void {
  const route = pathname.slice(ASSET_PREFIX.length);
  const separator = route.indexOf("/");
  const experimentId = separator >= 0 ? decodeSegment(route.slice(0, separator)) : undefined;
  const encodedPath = separator >= 0 ? route.slice(separator + 1) : "";
  const requestedPath = encodedPath
    .split("/")
    .map(decodeSegment)
    .filter((segment): segment is string => segment !== undefined)
    .join("/");
  if (!experimentId || !requestedPath) {
    sendJson(response, 404, { error: "Artifact not found." });
    return;
  }
  const filename = resolveReferencedAsset(projectRoot, experimentId, requestedPath);
  if (!filename) {
    sendJson(response, 404, { error: "Artifact not found." });
    return;
  }
  const contentType = mimeType(filename);
  response.statusCode = 200;
  response.setHeader("Content-Type", contentType);
  response.setHeader("Cache-Control", "no-store");
  if (!contentType.startsWith("image/") && contentType !== "application/pdf") {
    const safeName = path.basename(filename).replaceAll('"', "");
    response.setHeader("Content-Disposition", `attachment; filename="${safeName}"`);
  }
  createReadStream(filename).pipe(response);
}

async function readRequestJson(request: IncomingMessage): Promise<unknown> {
  let source = "";
  for await (const chunk of request) {
    source += chunk.toString();
    if (source.length > 8192) throw new Error("Request too large");
  }
  return JSON.parse(source);
}

async function handleRevealRequest(
  projectRoot: string,
  request: IncomingMessage,
  response: ServerResponse,
): Promise<void> {
  try {
    const body = await readRequestJson(request);
    if (
      typeof body !== "object" ||
      body === null ||
      !("experimentId" in body) ||
      !("reference" in body) ||
      typeof body.experimentId !== "string" ||
      typeof body.reference !== "string"
    ) {
      sendJson(response, 400, { error: "Invalid reveal request." });
      return;
    }
    const filename = resolveReferencedAsset(projectRoot, body.experimentId, body.reference);
    if (!filename) {
      sendJson(response, 404, { error: "Artifact not found." });
      return;
    }
    const { command, args } = revealCommandForPlatform(process.platform, filename);
    const child = spawn(command, args, { detached: true, stdio: "ignore" });
    child.on("error", () => undefined);
    child.unref();
    sendJson(response, 200, { revealed: true });
  } catch {
    sendJson(response, 400, { error: "Invalid reveal request." });
  }
}

function routeRequest(
  projectRoot: string,
  getIndex: () => SourceIndex,
  request: IncomingMessage,
  response: ServerResponse,
  next: () => void,
): void {
  if (!request.url) {
    next();
    return;
  }
  const pathname = new URL(request.url, "http://viewer.local").pathname;
  if (request.method === "POST" && pathname === REVEAL_ENDPOINT) {
    void handleRevealRequest(projectRoot, request, response);
    return;
  }
  if (request.method !== "GET") {
    next();
    return;
  }
  if (pathname === GRAPH_ENDPOINT) {
    sendJson(response, 200, snapshotFromIndex(getIndex()));
    return;
  }
  if (pathname.startsWith(DOCUMENT_PREFIX)) {
    const experimentId = decodeSegment(pathname.slice(DOCUMENT_PREFIX.length));
    const record = experimentId ? getIndex().experiments.get(experimentId) : undefined;
    if (!record) {
      sendJson(response, 404, { error: "Experiment not found." });
      return;
    }
    sendJson(response, 200, { id: record.id, title: record.title, body: record.body });
    return;
  }
  if (pathname.startsWith(ASSET_PREFIX)) {
    handleAssetRequest(projectRoot, pathname, response);
    return;
  }
  next();
}

export function autoresearchSource(projectRoot: string): Plugin {
  let index = scanProject(projectRoot);
  return {
    name: "autoresearch-project-source",
    configureServer(server: ViteDevServer) {
      const researchFile = path.join(projectRoot, "research.md");
      const experimentRoot = path.join(projectRoot, "research", "experiments");
      const framingRoot = path.join(projectRoot, "research", "framings");
      const watchedRoots = [researchFile, experimentRoot, framingRoot];
      server.watcher.add(watchedRoots);

      let refreshTimer: ReturnType<typeof setTimeout> | undefined;
      let pendingDocumentChange = false;
      let pendingAssetChange = false;
      const refresh = (changedPath: string) => {
        const change = sourceChangeForPath(
          index,
          researchFile,
          experimentRoot,
          framingRoot,
          changedPath,
        );
        if (!change.refresh) return;
        pendingDocumentChange ||= change.documentChanged;
        pendingAssetChange ||= change.assetChanged;
        if (refreshTimer) clearTimeout(refreshTimer);
        refreshTimer = setTimeout(() => {
          const documentChanged = pendingDocumentChange;
          const assetChanged = pendingAssetChange;
          pendingDocumentChange = false;
          pendingAssetChange = false;
          index = scanProject(projectRoot);
          server.ws.send({
            type: "custom",
            event: "autoresearch:update",
            data: { revision: index.revision, documentChanged, assetChanged },
          });
        }, 80);
      };
      server.watcher.on("add", refresh);
      server.watcher.on("change", refresh);
      server.watcher.on("unlink", refresh);
      server.watcher.on("addDir", refresh);
      server.watcher.on("unlinkDir", refresh);

      server.middlewares.use((request, response, next) => {
        routeRequest(projectRoot, () => index, request, response, next);
      });
    },
  };
}
