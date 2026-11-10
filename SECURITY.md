# Security Policy

## Supported versions

Security fixes are applied to the latest PyPI release of `free-llm-sdk` first.
Backports to older releases or pre-releases are not guaranteed (the project is Alpha).

## Reporting a vulnerability

Prefer a private report so keys and exploit details stay out of public issues:

1. Open a [new GitHub Security Advisory](https://github.com/physics91/freeaisdk/security/advisories/new)
   (or use [Security Advisories](https://github.com/physics91/freeaisdk/security/advisories) if private reporting is enabled).
2. Include impact, affected versions, and a minimal reproduction when possible.

If private reporting is disabled in the repository settings, do not file a public issue
with secrets or a working exploit; contact a maintainer first so reporting can be enabled.

## Out of scope

- Outages, rate limits, or model catalog drift on third-party free-tier providers.
- Caller-managed API keys and environment variables (the SDK never invents keys).
- Misconfigured deployments of `freeai serve`: current releases refuse non-loopback
  binds without a bearer token; forks or local patches that remove that check are
  outside this project’s security policy.
