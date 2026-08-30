set quiet

[windows]
set shell := ["powershell.exe", "-NoLogo", "-Command"]

stable := "stable"
msrv := "1.85.0"

# List every available development recipe.
default:
    just --list

# Install every Rust toolchain used to develop and verify the crate.
install: install-stable install-msrv

# Install stable Rust with the formatting and linting components.
install-stable:
    rustup toolchain install {{ stable }} --profile minimal --component rustfmt --component clippy

# Install the minimum Rust version supported by the crate.
install-msrv:
    rustup toolchain install {{ msrv }} --profile minimal

# Build an optimized version of the crate.
build:
    cargo +{{ stable }} build --workspace --all-features --release --locked

# Run the basic example as an optimized executable.
run:
    cargo +{{ stable }} run --release --locked --example basic

# Format every Rust target in the workspace.
format:
    cargo +{{ stable }} fmt --all

# Verify formatting without changing files.
format-check:
    cargo +{{ stable }} fmt --all -- --check

# Verify the Justfile's own formatting.
justfile-check:
    just --fmt --check

# Lint every target and deny warnings, including undocumented unsafe blocks.
lint:
    cargo +{{ stable }} clippy --workspace --all-targets --all-features --locked -- -D warnings -D clippy::undocumented_unsafe_blocks

# Build public documentation and deny rustdoc warnings.
docs $RUSTDOCFLAGS="-D warnings":
    cargo +{{ stable }} doc --workspace --all-features --no-deps --locked

# Run every documentation test.
doc-test:
    cargo +{{ stable }} test --workspace --doc --all-features --locked

# Run every test with the stable toolchain.
test:
    cargo +{{ stable }} test --workspace --all-targets --all-features --locked

# Run every test with the minimum supported Rust version.
test-msrv:
    cargo +{{ msrv }} test --workspace --all-targets --all-features --locked

# Run every test with compiler optimizations enabled.
test-release:
    cargo +{{ stable }} test --workspace --all-targets --all-features --release --locked

# Build and verify a package from a clean repository checkout.
package:
    cargo +{{ stable }} package --locked

# Build and verify a package while local changes are present.
package-dirty:
    cargo +{{ stable }} package --locked --allow-dirty

# Run the complete verification suite used before a pull request.
check: justfile-check format-check lint docs doc-test test test-msrv test-release package-dirty
