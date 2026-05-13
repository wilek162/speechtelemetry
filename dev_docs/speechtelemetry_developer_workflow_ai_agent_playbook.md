**speechtelemetry\
Developer Workflow & AI Agent Playbook**

*Version 0.1 \| May 2026*

# Purpose and scope

This document is the operational layer between the architecture
reference, the developer reference, and the library spec. It explains
how humans and AI agents should move from a requested change to a
correct, reviewed, testable implementation without breaking the
library-first, local-first design.

It does not redefine product scope, API contracts, or layer boundaries.
Those remain authoritative in the spec, architecture reference, and
developer reference. This playbook only tells contributors how to work
inside those rules.

# Source-of-truth order

  -----------------------------------------------------------------------
  Priority                            Use this for
  ----------------------------------- -----------------------------------
  1                                   Product intent, scope, and
                                      non-goals

  2                                   Layering, dependency direction, and
                                      extension rules

  3                                   Exact backend APIs, install
                                      commands, caveats, and defaults

  4                                   This workflow playbook for change
                                      execution and review

  5                                   Code, tests, fixtures, and
                                      generated artifacts
  -----------------------------------------------------------------------

# Working rules

-   Keep changes library-first. CLI and examples must never contain
    business logic that is absent from the library.

-   Keep dependencies flowing inward. Do not add imports from outer
    layers into inner layers.

-   Prefer fail-soft behavior. Return partial results with recorded
    errors instead of hard crashes when the architecture allows it.

-   Keep outputs explicit. Backend provenance, timing, and failure state
    belong in the result object, not in hidden logs.

-   Keep the runtime local-first. No new mandatory network calls,
    tokens, or external services unless the spec is updated first.

# Standard implementation loop

-   Identify the smallest change that solves the request. Write down the
    target stage, files, and expected output.

-   Check the spec, architecture reference, and developer reference for
    existing contracts, defaults, and constraints before editing code.

-   Edit the innermost layer first: types, interfaces, or backend code.
    Move outward only after the core contract is correct.

-   Update registry, orchestration, exporters, CLI, examples, and docs
    only when the change truly needs them.

-   Add or update tests before considering the change complete. A new
    backend or bug fix without tests is unfinished.

-   Run the project validation path that matches the change: unit tests,
    fixture-based integration tests, and any relevant benchmark checks.

-   Review the rendered or serialized output, not just the code diff.
    For document or export changes, inspect the actual artifact.

# Change playbooks

## Adding a new backend

-   Create the backend module in the correct stage directory.

-   Match the stage contract exactly, including method name, parameters,
    and return shape.

-   Guard optional imports and raise a clear availability error with
    installation guidance.

-   Register the backend in registry.py and add any required config
    option with a sensible default.

-   Add a fixture-backed test that exercises the backend through the
    public orchestration path.

## Adding a new output field

-   Update the canonical schema first so every consumer sees the same
    shape.

-   Add provenance and nullability explicitly; never encode missing data
    as a sentinel string.

-   Update exporters that should preserve or intentionally drop the
    field.

-   Add round-trip or schema tests so the field does not disappear in
    serialization.

## Changing orchestration or fail-soft behavior

-   Change core/pipeline.py only when the stage order or error policy
    truly changes.

-   Keep stage failures isolated and recorded. Do not let one failed
    optional stage stop unrelated stages.

-   Preserve the pre-flight checks for mandatory prerequisites such as
    FFmpeg and required tokens.

## Docs-only changes

-   Do not change code just to match wording unless the document exposes
    a real behavior mismatch.

-   Update the smallest document that owns the statement. Do not
    duplicate guidance across multiple docs unless the same rule appears
    in multiple contexts.

-   If a document disagrees with code, treat that as a defect and fix
    the mismatch explicitly.

# AI agent operating rules

-   Never invent an API, backend, model, or file path that is not
    already present in the source docs or repository.

-   When a change touches multiple layers, summarize the dependency
    chain before editing so the plan stays aligned with the
    architecture.

-   Prefer narrow edits over broad refactors. Preserve unrelated
    behavior unless the task explicitly asks for restructuring.

-   Call out uncertainty immediately. If the docs do not settle a point,
    stop and label the assumption instead of silently guessing.

-   Do not ask for extra clarification when the existing docs already
    provide a safe default. Make the best correct change that fits the
    documented contract.

# Definition of done

-   The code builds and the relevant tests pass.

-   The result objects and exported files reflect the documented schema.

-   Any new dependency, token, or license condition is documented in the
    correct reference.

-   The change does not violate layer boundaries or force a new runtime
    requirement.

-   A reviewer can understand what changed, why, and how it was verified
    without opening additional threads.

# PR handoff template

**Use this structure in the pull request description:**

-   What changed

-   Why it changed

-   Files touched

-   Tests run

-   Risks or follow-ups

-   Any dependency, license, or environment impact

# Recommended review order

1\. Verify the public contract: result schema, config defaults, and
backend names.

2\. Verify architecture: imports, layer boundaries, and registry
behavior.

3\. Verify behavior: tests, fixtures, and serialized outputs.

4\. Verify usability: error messages, docs, and examples.

# Close-out checklist

-   No hidden import-time side effects were introduced.

-   No required dependency or network call was added accidentally.

-   No architecture rule was bent to make the implementation easier.

-   No test was skipped without a documented reason.

-   The resulting artifact is ready for a developer or AI agent to use
    without guessing.
