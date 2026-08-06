<!--
Thanks for contributing to rai-sdk. Please read CONTRIBUTING.md if you have
not already: https://github.com/rmagatti/rai-sdk/blob/main/CONTRIBUTING.md
-->

## Description

<!-- What does this change do, and why? Focus on the motivation and the approach. -->

## Related issue

<!-- e.g. "Closes #123", "Fixes #123", or "N/A" for trivial changes. -->

Closes #

## Type of change

<!-- Keep the ones that apply, delete the rest. -->

- Bug fix (non-breaking change that fixes an issue)
- New feature (non-breaking change that adds functionality)
- Breaking change (existing code would need to be updated)
- Documentation only
- Tests only
- Build, CI, or tooling
- Refactor or performance (no behavior change)

## Checklist

- [ ] `cargo fmt --all --check` passes.
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` passes.
- [ ] `cargo test --all-features` passes, and every new test is deterministic and offline (no API credentials, no live provider calls).
- [ ] New or changed public items have rustdoc comments (`missing_docs` is enforced).
- [ ] `cargo doc --no-deps --all-features` passes with `RUSTDOCFLAGS="-D warnings"`.
- [ ] Feature-gated code still builds with `--no-default-features` and with each provider feature alone.
- [ ] Commit message(s) are clear and conventional-commit-formatted (release-plz turns them into the `CHANGELOG.md` entry automatically; no manual edit needed).
- [ ] The guide in `docs/` was updated if user-facing behavior changed.
- [ ] I agree to license my contribution under the terms of both `MIT` and `Apache-2.0`, as described in `CONTRIBUTING.md`.

## Notes for reviewers

<!-- Optional: tricky parts, open questions, follow-ups you deliberately left out. -->
