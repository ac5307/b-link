//! Concurrent ordered map and set collections implemented as a B-link tree.
//!
//! [`BLinkMap`] and [`BLinkSet`] use per-page read/write latches, two-sided
//! fences, right-sibling links, and forwarding pages. Readers correct stale
//! routes while concurrent writers split, rotate, merge, and unlink pages.
//!
//! Entry lookups return [`EntryRef`] or [`EntryMut`] guards. These guards make
//! borrowed access safe without cloning values, but they lock an entire leaf.
//! Drop them before re-entering the map for a potentially conflicting operation.
//!
//! ```
//! use b_link::{BLinkMap, BLinkSet};
//!
//! let map = BLinkMap::<u64, &str>::new();
//! assert_eq!(map.insert(7, "seven"), None);
//! assert_eq!(map.get(&7).map(|entry| *entry), Some("seven"));
//! assert_eq!(map.remove(&7), Some("seven"));
//!
//! let set = BLinkSet::<u64>::new();
//! assert!(set.insert(7));
//! assert!(set.contains(&7));
//! ```

#![warn(missing_docs)]
#![forbid(unsafe_op_in_unsafe_fn)]

mod array;
mod handle;
mod iter;
mod map;
mod node;
mod set;

pub use handle::{EntryMut, EntryRef};
pub use iter::{Iter, KeyRef, Keys, Range, ValueRef, Values};
pub use map::{BLinkMap, DEFAULT_PAGE_CAPACITY};
pub use set::{BLinkSet, SetIter, SetRange, SetRef};
