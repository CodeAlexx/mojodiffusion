# Serenity Canvas Contract

Status: binding for production changes under `serenity-server/canvas/`.

The canvas is the checked-in Serenity Studio browser client. Changes preserve
the existing Rust control-plane API, Mojo inference backends, workflow graph
schema, workflow builders, templates, model identifiers, and saved-workflow
compatibility unless a separately reviewed contract migration changes them.

## Source and generated artifacts

- JavaScript and its checked-in `.map` file are one versioned artifact. A
  change that alters generated JavaScript must preserve or regenerate its valid
  source map; source maps are not cleanup debris.
- `index.html`, CSS, assets, and JavaScript are served directly by the Rust
  server. Production behavior must not depend on an unrecorded local build
  directory, developer-home path, or another checkout.
- Existing browser globals and load order are compatibility surfaces. A module
  conversion or bundler migration requires an explicit migration and product
  gate.

## Product behavior

- Generate, queue, history, gallery, workflow, model, settings, and canvas
  state must reflect durable server state. The UI does not fabricate completed
  jobs, model availability, trainer artifacts, or cache readiness.
- Model-specific controls are driven by server capabilities and preserve the
  backend request contract. UI defaults may assist the user but must not
  silently change explicit request values.
- Failures remain visible and actionable. A missing server artifact is reported
  as missing; the browser must not require an external assistant to create it.
- User-selected output roots and run identifiers are treated as data from the
  server. The browser does not construct developer-machine filesystem paths.
- Prompt editors expose a practical multiline viewport and remain vertically
  resizable so longer prompts can be edited without relying on horizontal
  scrolling or a single-line-height control.

## Workflow canvas behavior

- Templates without an intentional saved layout are arranged as a
  left-to-right execution graph and fitted to the available viewport. Loading a
  workflow with saved coordinates preserves that arrangement while still
  fitting the complete graph on screen.
- Workflow execution status remains visible in the workflow header throughout
  submission, model and conditioning load, sampling progress, output decode and
  save, completion, interruption, and failure. Progress and completion state
  are driven by server job events rather than browser timers.
- The active execution path uses a persistent, high-contrast state: amber for
  active nodes and connections, green for completed work, and red for failures.
  Completion coloring remains until the next workflow execution begins.

## Verification

A production canvas change must pass the Rust server tests, JavaScript syntax
checks for touched files, source-map JSON validation for touched mapped files,
and a browser smoke covering the changed route. Inference changes additionally
require a decoded output artifact and completion evidence from the real worker.
