# Contributing

Please open an issue before substantial changes. Keep pull requests focused and
describe the affected behavior and reproduction steps.

Do not commit model weights, datasets, generated outputs, internal paths, or
credentials.

The first public commit is inference-only. Do not introduce training entry points,
training configurations, optimizer states, or copied third-party code without an
explicit scope and provenance review. Keep default-config changes consistent in both
`infer.yaml` and `anchor3r/configs/infer.yaml`.

Use `requirements.txt` for the pinned dependencies and see `docs/INSTALLATION.md`
for installation instructions.

The project website lives in a separate website checkout.
Do not add videos, demonstration clouds or page bundles under this repository's
`docs/`. See [Documentation and Repository Layout](README.md#documentation-and-repository-layout)
for the repository boundaries.
