# Security Policy

## Supported versions

`rai-sdk` is pre-1.0. Only the latest released version on
[crates.io](https://crates.io/crates/rai-sdk) receives security fixes, and fixes
are shipped as a new patch or minor release rather than backported to older
lines.

| Version | Supported |
| --- | --- |
| 0.1.x | Yes — the current release line |
| < 0.1 | No |

Once `1.0.0` is released this policy will be revisited to cover the `1.x` line
explicitly.

## Reporting a vulnerability

**Do not report security vulnerabilities through public GitHub issues, pull
requests, or discussions.**

Preferred: use GitHub's private vulnerability reporting.

1. Go to <https://github.com/rmagatti/rai-sdk/security/advisories/new>.
2. Fill in the report. Only the maintainers can see it.

If you cannot use GitHub private reporting, email
[ronniemagatti@gmail.com](mailto:ronniemagatti@gmail.com) with `rai-sdk security`
in the subject line.

### What to include

The more of this you can provide, the faster a fix lands:

- The affected version of `rai-sdk` and the enabled Cargo features.
- A description of the vulnerability and its impact.
- Steps to reproduce, ideally a minimal Rust snippet.
- Which provider (OpenAI, Anthropic, OpenRouter) is involved, if any.
- Any known mitigations or workarounds.

> [!CAUTION]
> **Never include API keys, tokens, or other credentials in a report**, in any
> form — not in logs, backtraces, HTTP captures, screenshots, or code. Redact
> anything resembling `sk-...`, `sk-ant-...`, or `sk-or-...` before sending. If
> you believe a credential of yours has been exposed, revoke and rotate it with
> the provider immediately; that is outside this project's control.

## What to expect

- **Acknowledgement** within 5 business days.
- **An initial assessment**, including whether the report is accepted as a
  vulnerability and a rough severity, within 10 business days.
- **Progress updates** at least every 14 days while the report is open.
- **Disclosure**: once a fix is released, a GitHub Security Advisory is
  published with a CVE where appropriate, and the `CHANGELOG.md` entry notes the
  fix. You will be credited unless you ask otherwise.

This is a small, volunteer-maintained project, so please treat these as
good-faith targets rather than a contractual SLA. Please allow a reasonable
period for a fix before disclosing publicly.

## Scope

In scope:

- Vulnerabilities in this crate's own code: request construction, response and
  SSE parsing, tool argument handling, schema validation, credential handling,
  and error paths that could leak sensitive data.
- Dependency vulnerabilities that are reachable through `rai-sdk`'s public API.

Out of scope:

- Vulnerabilities in OpenAI, Anthropic, or OpenRouter themselves. Report those
  to the relevant provider.
- Model behavior concerns such as prompt injection, jailbreaks, or harmful
  generated content. These are properties of the models, not of this SDK.
- Issues that require an attacker to already control the host process or its
  environment variables.
- Leaked credentials in your own application, repository, or logs.

## Handling credentials safely

`rai-sdk` reads API keys from environment variables (`OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) or from explicit `ClientBuilder`
calls, and sends them only to the configured provider base URL. When using this
crate:

- Keep keys in environment variables or a secret manager, never in source
  control.
- Be careful with custom base URLs — pointing a client at an untrusted host sends
  your credentials there.
- If you log `Config` or request state, make sure your logging does not capture
  key material.
