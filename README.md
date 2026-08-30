# b-link

`b-link` provides `BLinkMap` and `BLinkSet`, concurrent ordered collections
backed by a latch-crabbed B-link/B+ tree. Pages use this project’s own
fixed-capacity contiguous `Array<T>`; the implementation does not use `Vec` for
page storage or tree algorithms.

```rust
use b_link::{BLinkMap, BLinkSet};

let map = BLinkMap::<u64, String>::new();
map.insert(7, String::from("seven"));
assert_eq!(map.get(&7).unwrap().value(), "seven");

let set = BLinkSet::<u64>::new();
assert!(set.insert(7));
assert!(set.contains(&7));
```

## Layout

- `map.rs` is the public `BLinkMap` facade.
- `set.rs` implements `BLinkSet` on zero-sized map values.
- `node.rs` owns page types and every tree algorithm: traversal, split,
  rotation, merge, unlink, root publication, and invariant validation.
- `handle.rs` provides `EntryRef` and `EntryMut` leaf-latch guards.
- `iter.rs` provides guarded map/set iterator machinery and ranges.
- `array.rs` is the unsafe, fixed-capacity contiguous allocation used for
  records, separators, and child pointers.

## Page and routing invariants

Every active page owns a half-open interval `[low_key, high_key)`. Missing
fences mean negative or positive infinity. Internal separators are inclusive
lower bounds for the child to their right, and a page with `n` separators has
exactly `n + 1` children.

Pages at each level form a forward sibling chain. If a stale traversal reaches
a page above its high fence, it moves right. If a deletion rotation moved the
boundary left and the key is now below the page’s lower fence, traversal
restarts from the current root. A merged page becomes a rightward forwarding
page before its parent and sibling pointers are removed, so an `Arc` cloned by
an earlier traversal remains safe until it drains.

Non-root leaves contain at least `N / 2` records. Non-root internal pages
contain at least `(N - 1) / 2` separators. These minima ensure two minimally
occupied siblings plus an internal separator always fit in one `N`-slot page
for odd and even `N`.

## Concurrency

There is no global structure lock. Reads and ordinary inserts/removes latch one
leaf. Structural writers descend top-down and briefly latch a parent plus its
selected child and, for deletion, one adjacent sibling. Siblings are always
latched left-to-right. A full child is split before descent; a minimally
occupied child borrows or merges before descent. The parent latch is released
as soon as the child is safe, allowing writers in disjoint subtrees to proceed
concurrently.

The root slot is latched only while a full root is replaced or an empty root
level is collapsed. Root-role state lives under each page latch so no operation
reacquires the root-slot latch from below.

`EntryRef`, `EntryMut`, iterator items, and set references lock an entire leaf.
Drop guards before conflicting operations on that leaf. Retaining guards while
re-entering the collection can self-deadlock, and acquiring several guards in
inconsistent orders can form a cross-thread cycle.

## Iteration and ranges

`iter`, `keys`, `values`, and `range` return ascending guarded cursors. They are
weakly consistent under concurrent mutation: a raced insert/remove may or may
not appear, but keys never regress or repeat. Each step re-seeks strictly after
the last shared key handle, so page splits, merges, and rotations do not leave
stale array indices in the iterator.

Ranges accept borrowed bounds when `K: Borrow<Q>` and the `K`/`Q` orderings
agree. As with Rust’s standard ordered collections, an inverted range or equal
excluded endpoints panic.

## Capacity and storage

The const parameter `N` is the exact record/separator allocation capacity of a
page. `N` must be at least 3; omitting it uses `DEFAULT_PAGE_CAPACITY` (32).
Internal pages allocate exactly `N + 1` child pointers. Keys are shared through
`Arc`, so separators and fences do not require `K: Clone`.

Because those metadata handles share key ownership, removing a record returns
its value immediately but a key allocation still serving as a separator or
fence can remain alive until that boundary is replaced or the collection drops.

## Verification

The suite includes deterministic and randomized differential traces against
`BTreeMap` from the Rust Standard Library, odd/even page capacities, complete
deletion and root collapse, borrowed lookups/ranges, public API compilation,
set behavior, and concurrent read/write/split/rotate/merge stress. Install
`just` using the platform instructions in
[CONTRIBUTING.md](https://github.com/ac5307/b-link/blob/master/CONTRIBUTING.md#development-workflow),
then run:

```text
just install
just check
```

The cross-platform
[Justfile](https://github.com/ac5307/b-link/blob/master/Justfile) runs the same
focused commands used by CI. Run `just` to list its build, run, format, lint,
documentation, test, and packaging recipes.

## Contributing

Contributions are welcome. Read
[CONTRIBUTING.md](https://github.com/ac5307/b-link/blob/master/CONTRIBUTING.md)
before submitting a change, include tests and documentation where appropriate,
and run `just check` before opening a pull request.

## Code of Conduct

Everyone participating in this project must follow the
[Code of Conduct](https://github.com/ac5307/b-link/blob/master/CODE_OF_CONDUCT.md).
Please help keep the community respectful, constructive, and welcoming.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or
  <https://opensource.org/licenses/MIT>)

at your option.
