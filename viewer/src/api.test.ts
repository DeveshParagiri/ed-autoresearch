import { describe, expect, it } from "vitest";

import { isImageReference } from "./api";

describe("image references", () => {
  it("recognizes linked figures independently of host or project path", () => {
    expect(isImageReference("figures/01-score-summary.png")).toBe(true);
    expect(isImageReference("runs/run.test/model.svg?v=2#panel-a")).toBe(true);
    expect(isImageReference("https://example.org/result.WEBP?download=1")).toBe(true);
    expect(isImageReference("data:image/png;base64,AA==")).toBe(true);
  });

  it("leaves non-image evidence to the ordinary artifact action", () => {
    expect(isImageReference("runs/run.test/metrics.json")).toBe(false);
    expect(isImageReference("figures/README.md")).toBe(false);
    expect(isImageReference("#evidence")).toBe(false);
  });
});
