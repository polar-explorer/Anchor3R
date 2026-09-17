# Security Policy

Report suspected vulnerabilities privately to Peilin Tao at
[taopeilin2023@ia.ac.cn](mailto:taopeilin2023@ia.ac.cn). The maintainer has confirmed
this address for both project feedback and private security reports.

Use a subject such as `[Anchor3R Security] Brief description` and include the
affected version or commit, impact, and minimal reproduction steps when available.
Do not include credentials, private images, proprietary checkpoints, or other
sensitive datasets; use a sanitized example instead. Do not disclose vulnerability
details in public issues.

GitHub private vulnerability reporting has not been verified as enabled. Use the
email address above rather than assuming that a GitHub reporting channel exists.

## Checkpoints

Only load weights from an approved source and verify the release SHA256.
Inference uses PyTorch weights-only loading; this is not a guarantee that any
untrusted file is harmless. The release does not provide an unsafe pickle-loading
fallback or conversion back into a training checkpoint.

## Viewer

The viewer is intended for local/trusted use and binds to `127.0.0.1` by default.
It has no application-level authentication. For remote access, keep the default
binding and create a tunnel from your local computer:

```bash
ssh -L 8080:127.0.0.1:8080 user@your-server
```

Then open `http://127.0.0.1:8080` locally. Only use `--host 0.0.0.0` behind an
appropriate access-controlled network/proxy. Do not expose sensitive scene data
through a public port.

## Release Hygiene

Never commit credentials, `.env` files, training checkpoints, datasets, or private
paths. Ignore rules are preventative, not a Git-history scan. Audit the actual
staged files and complete history before making a repository public. Rotate any
credential that has already been exposed; merely deleting its file is not enough.
