# Contributing

Contributions are welcome. The authoritative guide lives in the repository:

- [CONTRIBUTING.md](https://github.com/rmagatti/rai-sdk/blob/main/CONTRIBUTING.md) — setup, workflow, code style, and PR expectations
- [CODE_OF_CONDUCT.md](https://github.com/rmagatti/rai-sdk/blob/main/CODE_OF_CONDUCT.md)
- [SECURITY.md](https://github.com/rmagatti/rai-sdk/blob/main/SECURITY.md) — report vulnerabilities privately, never in a public issue

## What CI enforces

```sh
cargo fmt --all --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test --doc --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Two policies are worth calling out because they surprise people:

- **`missing_docs` is enforced.** Every new public item needs documentation, or the build fails.
- **Tests must be offline.** The suite runs with no provider credentials and must never make live API calls. Provider behavior is tested against a mock HTTP server through the base-URL configuration. CI actively fails if credentials are present.

The crate also sets `unsafe_code = "forbid"`, and the MSRV (currently 1.86) is read from `Cargo.toml` and verified in CI.

## Working on this guide

The guide is an [mdBook](https://rust-lang.github.io/mdBook/) in `docs/`:

```sh
cargo install mdbook --locked
mdbook serve docs --open   # live reload while editing
mdbook build docs          # one-off build
```

Add a chapter by creating the file in `docs/src/` and listing it in `docs/src/SUMMARY.md` — a page missing from `SUMMARY.md` will not appear in the book.

Pushes to `main` deploy the guide to GitHub Pages automatically. Pull requests build it without deploying, so a broken book is caught in review.

## Reporting issues

Use the issue templates. For bugs, include the `rai-sdk` version, Rust version, OS, provider, and enabled features — provider-specific and feature-gating bugs are common and hard to reproduce without those details.

**Never paste API keys** into an issue, including inside logs or backtraces.
