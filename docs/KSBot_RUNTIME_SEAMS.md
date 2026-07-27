# KSBot runtime seams

This fork keeps one KSBot patch commit on top of upstream GenericAgent. KSBot pins
that commit by SHA; the human-readable release tag identifies the matching baseline.

The patch owns only reusable GenericAgent seams:

- `GA_MEMORY_ROOT` for externally owned long-term memory;
- `execute_task()` with native history, working state, loop, terminal result, interrupt,
  abort and cleanup semantics;
- common long-input preparation for native task entry points;
- the actual tool cwd in the system context;
- a stateless `model_call()` using the active GA model configuration;
- process-wide inline-eval cwd and memory-settlement serialization;
- an unambiguous `code_run` schema for Python source versus shell commands.

## Memory lifecycle

When `GA_MEMORY_ROOT` is unset, GenericAgent uses the checkout's bundled `memory/`.
When it is set, the external root is wholly user-owned mutable memory: L1/L2, L3 SOPs,
helper scripts, indexes and future files may all be changed by the Agent. On startup,
bundled files missing from the external root are copied as initial defaults. Existing
external files are never overwritten or upgraded automatically.

Therefore a GA upgrade may add new defaults, but changes to an already initialized
SOP, template or helper require an explicit reviewed migration. Replacing the vendor
checkout must never delete or rewrite the external memory root.

## Rebase procedure

1. Fetch upstream and rebase the single KSBot patch commit onto the selected revision.
2. Resolve changes at the GenericAgent owner seam, not in the KSBot product adapter.
3. Run `tests/test_execute_task.py` and `tests/test_memory_paths.py`.
4. Run the KSBot contract probe and full KSBot test suite against the new SHA.
5. Amend the one patch commit, force-push with lease, then update KSBot `GA_REVISION`.

Do not add WPS transport, Kubernetes policy or product prompts to this fork.
