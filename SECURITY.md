# Security Policy

## Reporting a vulnerability

Please report security issues **privately**. Do not open a public issue for a
vulnerability or an exposed credential.

- Preferred: open a private advisory via
  **GitHub → Security → Report a vulnerability**
  (<https://github.com/ianjarog/galaxydisksize/security/advisories/new>).
- Alternatively, email the maintainer (see `CITATION.cff`).

You will receive an acknowledgement as soon as possible. Please include enough
detail to reproduce the issue.

## Handling secrets

This repository must never contain credentials. In particular:

- No passwords, API keys, personal access tokens, SSH private keys, or
  `.netrc` / `.pypirc` files.
- Database and service credentials are read from **environment variables** at
  run time (see `.env.example`); they are never hard-coded.
- `.env` and other local secret files are git-ignored.

If you discover a committed secret:

1. Treat the credential as **compromised** and rotate it at the source
   immediately — removing it from the repository does not un-leak it.
2. Report it privately using the channel above.
3. The maintainer will purge it from the working tree and, where appropriate,
   from history before any further release.

## Supported versions

This is research software released alongside a publication. Security fixes are
applied to the latest released version only.
