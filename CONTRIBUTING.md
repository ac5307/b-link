# Contributing to b-link

Thank you for helping improve `b-link`.

## Before submitting a change

1. Discuss substantial API or concurrency-model changes in an issue first.
2. Keep page storage based on the crate's `Array<T>` rather than `Vec`.
3. Keep tree algorithms in `node.rs` and public collection APIs in their
   corresponding facade modules.
4. Add focused tests for behavioral changes, especially split, merge, range,
   iterator, and concurrency behavior.
5. Update public documentation when an API or observable behavior changes.

## Development workflow

Development commands live in the cross-platform [Justfile](Justfile). Install
[Rustup](https://rustup.rs/) first, then install `just` with the command for
your platform:

| Platform | Command |
| --- | --- |
| Linux | `cargo install just` |
| macOS | `brew install just` |
| Windows | `winget install --id Casey.Just --exact` |

Other supported installation methods are listed in the
[`just` package guide](https://just.systems/man/en/packages.html).
The Justfile uses PowerShell automatically on Windows, so Git Bash is not
required.

Run `just install` to install stable Rust with Rustfmt and Clippy plus the
minimum supported Rust version used by CI. Then use `just` to list every
recipe. The most common commands are:

```text
just build         # optimized build
just run           # optimized basic example
just test          # stable test suite
just test-release  # optimized test suite
just check         # complete pre-pull-request verification
```

Run `just check` before opening a pull request. Use clear commit messages;
Conventional Commit prefixes such as `feat:`, `fix:`, `perf:`, and `docs:`
produce better release notes.

By participating, you agree to follow the project's
[Code of Conduct](CODE_OF_CONDUCT.md).

## Licensing

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in `b-link` is licensed under the same terms as the
project, without additional terms or conditions.
