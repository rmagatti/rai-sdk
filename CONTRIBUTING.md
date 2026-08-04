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
cargo test --no-default-features --features openai
cargo test --no-default-features --features anthropic
cargo test --no-default-features --features openrouter
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

The `test` job runs on `ubuntu-latest` across `--all-features`,
`--no-default-features`, and each provider feature alone, plus `--all-features`
on `macos-latest`.

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

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Make your change, with tests for anything behavioral.
3. Run `cargo fmt --all`, clippy, and the tests.
4. Add an entry under `## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md) for
   anything user-visible.
5. Open the pull request against `main` and fill in the template.

Guidelines:

- One logical change per pull request. Unrelated refactors make review harder.
- Keep the description focused on motivation and approach; the diff shows the
  rest.
- Note explicitly if the change is breaking, and describe the migration.
- Expect review feedback; push follow-up commits rather than force-pushing over
  history that has already been reviewed, unless asked.
- All CI jobs must be green before merge.

## Releases

Releases are cut by a maintainer: the version in `Cargo.toml` is bumped, the
`Unreleased` changelog section is promoted to a versioned section, and a `vX.Y.Z`
tag is pushed. The tag triggers
[`.github/workflows/release.yml`](.github/workflows/release.yml), which verifies
the tree and publishes to crates.io. Contributors do not need to do anything
release-related beyond keeping the changelog current.

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
