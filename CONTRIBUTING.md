# Contributing to rai-sdk

Thanks for your interest in improving `rai-sdk`. This document covers how to
build and test the project, what CI enforces, and the conventions used for
commits and pull requests.

By participating in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

- For **bugs**, open an issue with a minimal reproduction.
- For **new features or API changes**, open an issue first. `rai-sdk` is pre-1.0
  and the public API is still settling, so it is much cheaper to agree on a
  design before code is written.
- For **security vulnerabilities**, do not open a public issue. Follow
  [SECURITY.md](SECURITY.md).
- Small fixes (typos, docs, obvious bugs) can go straight to a pull request.

## Prerequisites

- A Rust toolchain at or above the MSRV declared as `rust-version` in
  [`Cargo.toml`](Cargo.toml) (currently **1.86**). `rustup` is recommended.
- The `rustfmt` and `clippy` components:
  ```sh
  rustup component add rustfmt clippy
  ```
- Optionally [mdBook](https://rust-lang.github.io/mdBook/) if you want to work
  on the guide:
  ```sh
  cargo install mdbook
  ```

No API keys or accounts are needed to build, test, or document the project.

## Build and test

```sh
# Clone and build.
git clone https://github.com/rmagatti/rai-sdk.git
cd rai-sdk
cargo build --all-features

# Run the test suite, including doctests.
cargo test --all-features
```

Every provider is behind a Cargo feature, so it is easy to break feature gating
without noticing. Check the combinations CI checks:

```sh
cargo test --no-default-features
cargo test --no-default-features --features openai,rustls-tls
cargo test --no-default-features --features anthropic,rustls-tls
cargo test --no-default-features --features openrouter,rustls-tls
cargo test --no-default-features --features openai,native-tls
```

### Examples

The bundled examples in [`examples/`](examples) are real programs and do call
providers, so they need credentials in your environment. They are not part of
the test suite and CI never runs them:

```sh
export OPENAI_API_KEY="sk-..."
cargo run --example basic_chat
```

## Testing policy: offline and deterministic

**Tests must never make live API calls and must never require credentials.**

- Do not read real provider keys in tests, and do not skip tests when a key is
  absent. A test that only runs when someone has an API key is a test that never
  runs in CI.
- Mock the provider HTTP surface instead. The dev-dependency
  [`wiremock`](https://docs.rs/wiremock) is available for this: stand up a local
  mock server and point the client at it with `ClientBuilder::openai_base_url`,
  `anthropic_base_url`, or `openrouter_base_url`.
- Tests that mutate process-wide state (notably environment variables read by
  `Config::from_env`) must be serialized with
  [`serial_test`](https://docs.rs/serial_test)'s `#[serial]` attribute, because
  the test harness runs tests in parallel threads within one process.
- Avoid wall-clock sleeps and real timing assumptions. Assert on computed
  backoff values rather than on elapsed time.
- Tests must be deterministic: no reliance on network availability, hash
  iteration order, or randomness without a fixed seed.

CI runs with no provider environment variables set and actively fails if any of
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `OPENROUTER_API_KEY` is present.

## Code style and lints

Formatting and lints are not negotiable in CI, so run them locally before
pushing:

```sh
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
```

Additional expectations:

- **Public items need documentation.** `Cargo.toml` sets
  `[lints.rust] missing_docs = "warn"`, and the CI docs job denies warnings, so
  an undocumented public item, module, field, or enum variant will fail the
  build. Write a rustdoc comment for anything you make public.
- **`unsafe` is forbidden.** `[lints.rust] unsafe_code = "forbid"` is set at the
  manifest level.
- **Intra-doc links must resolve.** `broken_intra_doc_links` is denied.
- Doc examples should compile. Use ```` ```no_run ```` for examples that build a
  client or issue a request, since those would otherwise try to reach a provider
  when doctests run.
- Respect the MSRV. Do not use language or standard library features newer than
  the declared `rust-version`; if a change genuinely requires a newer compiler,
  say so in the pull request so the MSRV bump can be a deliberate decision.
- Match the surrounding style: builder methods return `Self`, fallible paths
  return the crate's `Result`, and diagnostics go through `tracing` rather than
  `println!`.

## Documentation

There are two documentation surfaces:

- **API reference** — rustdoc in `src/`, published to
  [docs.rs](https://docs.rs/rai-sdk). Build it locally with:
  ```sh
  RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --open
  ```
- **Guide** — an mdBook in [`docs/`](docs), published to
  <https://rmagatti.github.io/rai-sdk/>. Build or preview it with:
  ```sh
  mdbook build docs
  mdbook serve docs   # serves at http://localhost:3000 with live reload
  ```

If your change alters user-facing behavior, update both the rustdoc and the
relevant guide chapter.

## Exactly what CI runs

CI is defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) and runs
on pushes to `main` and on pull requests. Reproducing these locally means very
few surprises:

| Job | Command |
| --- | --- |
| `fmt` | `cargo fmt --all --check` |
| `clippy` | `cargo clippy --all-targets --all-features --locked -- -D warnings` |
| `test` | `cargo test --locked <features> --all-targets` then `cargo test --locked <features> --doc` |
| `docs` | `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features --locked` |
| `msrv` | `cargo check --all-features --locked` on the toolchain declared as `rust-version` |
| `package` | `cargo package --list --locked` then `cargo publish --dry-run --locked` |

The `test` job runs on `ubuntu-latest` across `--all-features`, a providerless
`--no-default-features` build, each provider with rustls, and OpenAI with
native-tls, plus `--all-features` on `macos-latest`.

`Cargo.lock` is committed and CI uses `--locked`. If your change requires a
dependency update, commit the resulting `Cargo.lock` along with it.

## Commit conventions

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <short imperative summary>

<optional body explaining what and why>
```

Types in use: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`,
`build`. Useful scopes: `openai`, `anthropic`, `openrouter`, `client`, `tool`,
`stream`, `retry`, `model`, `config`.

Examples:

```
fix(anthropic): stop dropping the final SSE chunk on tool use
docs(readme): document the openrouter attribution headers
```

Keep commits focused, and write a body when the reasoning is not obvious from
the diff. Mark breaking changes with a `!` after the type (`feat!:`) and explain
the migration in the body.

**Pull request titles matter beyond style here.** This repository merges pull
requests by squash only, so your PR title becomes the single commit message on
`main` — and that commit message is exactly what release-plz reads to decide
whether to cut a release and how to categorize it in `CHANGELOG.md`. Give the
PR the same conventional-commit-formatted title you'd give a single commit. CI
enforces this (the `conventional-title` job in
[`ci.yml`](.github/workflows/ci.yml)) and it is a required check.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests for anything behavioral.
3. Run `cargo fmt --all`, clippy, and the tests.
4. Open the pull request against `main`, with a conventional-commit-formatted
   title (see above), and fill in the template.

You do not need to touch `CHANGELOG.md` yourself: release-plz generates its
entry from your PR title/commit message when your PR is squash-merged (see
"Releases" below).

Guidelines:

- One logical change per pull request. Unrelated refactors make review harder.
- Keep the description focused on motivation and approach; the diff shows the
  rest.
- Note explicitly if the change is breaking, and describe the migration.
- Expect review feedback; push follow-up commits rather than force-pushing over
  history that has already been reviewed, unless asked.
- All required checks must be green before merge, enforced by this
  repository's branch ruleset on `main` (which also disallows force pushes and
  branch deletion) — see the maintainer setup checklist in this repository's
  release-automation pull request if the ruleset isn't showing up yet under
  Settings -> Rules.
- Merging is by squash only; merge commits and rebase merges are disabled on
  this repository so the PR title reliably becomes the commit release-plz
  reads.

## Releases

Releases are automated end to end, with **no Release PR and no separate ship
decision**: merging a PR to `main` is the release decision. This is a
deliberate trade-off, chosen over a Release-PR flow, in exchange for shipping
immediately instead of batching changes behind a second PR:

1. Merging a PR (squash-merges it, using the PR title as the commit message)
   pushes that commit to `main`, which runs
   [`.github/workflows/release-plz.yml`](.github/workflows/release-plz.yml).
2. That workflow checks whether any commit since the last tag is a `feat`,
   `fix`, `perf`, `refactor`, `revert`, or breaking (`!`/`BREAKING CHANGE`)
   commit. If not — e.g. a `docs`, `chore`, `ci`, `test`, `style`, or `build`
   only change — it stops here. Nothing is released.
3. Otherwise it runs `release-plz update`, which bumps the version in
   `Cargo.toml` and adds a `CHANGELOG.md` entry generated from conventional
   commit messages since the last release, then commits that directly to
   `main` as `chore(release): vX.Y.Z`, and creates and pushes the
   corresponding `vX.Y.Z` tag. release-plz itself never runs `cargo publish`
   and never creates the GitHub Release (see
   [`release-plz.toml`](release-plz.toml)).
4. That tag triggers
   [`.github/workflows/release.yml`](.github/workflows/release.yml), which
   re-verifies the tree (fmt, clippy, tests, the tag-matches-`Cargo.toml`
   check), runs the one and only `cargo publish` to crates.io, and creates the
   GitHub Release.

**The accepted trade-off:** every `feat`/`fix`/`perf`/`refactor`/`revert` (or
breaking) merge to `main` publishes to crates.io immediately — there is no
window to batch several merges into one release, and no maintainer review step
between "PR merged" and "published." crates.io has no unpublish, only
[yank](https://doc.rust-lang.org/cargo/reference/publishing.html#cargo-yank);
a bad release is fixed forward (a follow-up patch release) or yanked, never
erased. Reviewing the PR before merge — the same review every PR already
gets — is the only gate.

Contributors do not need to do anything release-related beyond writing a
conventional-commit-formatted PR title; see "Pull requests" above.
`.github/workflows/release.yml` requires a `CARGO_REGISTRY_TOKEN` repository
secret; without it, `verify` still runs but `publish` fails immediately with an
explanatory message, so nothing is ever half-published.
`.github/workflows/release-plz.yml` pushes directly to `main` and pushes tags,
so this repository's branch ruleset on `main` needs to list the
`github-actions` app as an actor exempt from the ruleset's
required-status-checks rule — otherwise a brand new commit that was never
itself pushed to a checked ref would fail that same requirement it's meant to
satisfy. See the comments at the top of that workflow for details, including
the self-trigger guard that stops it from reacting to its own commit.

## License

`rai-sdk` is dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <http://opensource.org/licenses/MIT>)

at your option.

### Contribution terms

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
