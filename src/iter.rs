use std::{
  borrow::Borrow,
  cmp::Ordering,
  fmt,
  iter::FusedIterator,
  marker::PhantomData,
  ops::{Bound, Deref, RangeBounds},
};

use crate::{
  handle::KeyHandle,
  map::{BLinkMap, DEFAULT_PAGE_CAPACITY},
  node::{NodeKind, NodeReadGuard, Tree},
  EntryRef,
};

/// An immutable, ascending iterator over a [`BLinkMap`].
///
/// Each yielded [`EntryRef`] read-locks its containing leaf. The iterator keeps
/// only a shared handle to the last key it returned and re-seeks strictly after
/// that key on the next call. Consequently page splits, merges, and rotations
/// cannot make the iterator move backward or yield one key twice.
///
/// This is a weakly consistent concurrent iterator: it observes every record
/// that remains present while the cursor passes it, but a record inserted or
/// removed concurrently may or may not be observed. It never yields keys out
/// of order.
///
/// # Locking
///
/// A yielded guard may be retained after the next call, but doing so keeps its
/// entire leaf locked. Drop guards promptly. In particular, calling `next`
/// while retaining a previous item from the same leaf can wait behind a queued
/// writer, which is itself waiting for that retained item.
pub struct Iter<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Tree being traversed.
  tree: &'map Tree<K, V, N>,
  /// Last returned key; its `Arc` survives even if that record is removed.
  last: Option<KeyHandle<K>>,
  /// Once exhausted, concurrent inserts do not revive this iterator.
  finished: bool,
}

impl<'map, K, V, const N: usize> Iter<'map, K, V, N> {
  /// Creates a cursor before the first key.
  pub(crate) fn new(map: &'map BLinkMap<K, V, N>) -> Self {
    Self {
      tree: &map.tree,
      last: None,
      finished: false,
    }
  }
}

impl<'map, K: Ord, V, const N: usize> Iterator for Iter<'map, K, V, N> {
  type Item = EntryRef<'map, K, V, N>;

  /// Seeks the least key strictly greater than the previous result.
  fn next(&mut self) -> Option<Self::Item> {
    if self.finished {
      return None;
    }

    let candidate = match &self.last {
      None => self.tree.first_entry(),
      Some(last) => self.tree.entry_at_or_after(last.get(), false),
    };
    let Some((guard, index)) = candidate else {
      self.finished = true;
      return None;
    };

    self.last = Some(clone_key(&guard, index));
    Some(EntryRef::new(guard, index))
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    // Concurrent insertions mean no stable finite upper bound can be promised.
    (0, None)
  }
}

impl<K: Ord, V, const N: usize> FusedIterator for Iter<'_, K, V, N> {}

/// An immutable ascending iterator over a bounded key range.
///
/// Items and concurrency semantics are the same as for [`Iter`]. The supplied
/// range object is stored by value, so both owned bounds and borrowed bounds
/// are supported without cloning `K` or the borrowed lookup type `Q`.
pub struct Range<
  'map,
  K,
  V,
  Q: ?Sized,
  R,
  const N: usize = DEFAULT_PAGE_CAPACITY,
> {
  /// Tree being traversed.
  tree: &'map Tree<K, V, N>,
  /// User-provided start and end bounds.
  bounds: R,
  /// Last key returned after the initial bound lookup.
  last: Option<KeyHandle<K>>,
  /// Whether the start bound has already been applied.
  started: bool,
  /// Sticky exhaustion marker.
  finished: bool,
  /// Records the borrowed lookup type without owning one.
  _query: PhantomData<fn(&Q)>,
}

impl<'map, K, V, Q: Ord + ?Sized, R, const N: usize>
  Range<'map, K, V, Q, R, N>
where
  R: RangeBounds<Q>,
{
  /// Validates and stores a range cursor.
  pub(crate) fn new(map: &'map BLinkMap<K, V, N>, bounds: R) -> Self {
    validate_bounds::<Q, R>(&bounds);
    Self {
      tree: &map.tree,
      bounds,
      last: None,
      started: false,
      finished: false,
      _query: PhantomData,
    }
  }
}

impl<'map, K, V, Q, R, const N: usize> Iterator
  for Range<'map, K, V, Q, R, N>
where
  K: Ord + Borrow<Q>,
  Q: Ord + ?Sized,
  R: RangeBounds<Q>,
{
  type Item = EntryRef<'map, K, V, N>;

  /// Applies the start bound once, then re-seeks after each returned key.
  fn next(&mut self) -> Option<Self::Item> {
    if self.finished {
      return None;
    }

    let candidate = if !self.started {
      self.started = true;
      match self.bounds.start_bound() {
        Bound::Unbounded => self.tree.first_entry(),
        Bound::Included(start) => self.tree.entry_at_or_after(start, true),
        Bound::Excluded(start) => self.tree.entry_at_or_after(start, false),
      }
    } else {
      let last = self
        .last
        .as_ref()
        .expect("a started, unfinished range has returned a key");
      self.tree.entry_at_or_after::<K>(last.get(), false)
    };

    let Some((guard, index)) = candidate else {
      self.finished = true;
      return None;
    };

    let key = clone_key(&guard, index);
    if !inside_end_bound::<K, Q, R>(&key, &self.bounds) {
      self.finished = true;
      return None;
    }

    self.last = Some(key);
    Some(EntryRef::new(guard, index))
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    (0, None)
  }
}

impl<K, V, Q, R, const N: usize> FusedIterator for Range<'_, K, V, Q, R, N>
where
  K: Ord + Borrow<Q>,
  Q: Ord + ?Sized,
  R: RangeBounds<Q>,
{
}

/// A read guard exposing only the key half of one map record.
#[must_use = "the key's leaf remains read-locked while this guard is held"]
pub struct KeyRef<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Full record guard supplying the page latch and key reference.
  entry: EntryRef<'map, K, V, N>,
}

impl<K, V, const N: usize> KeyRef<'_, K, V, N> {
  /// Wraps a full entry guard without acquiring another latch.
  fn new(entry: EntryRef<'_, K, V, N>) -> KeyRef<'_, K, V, N> {
    KeyRef { entry }
  }

  /// Returns the guarded key.
  pub fn get(&self) -> &K {
    self.entry.key()
  }
}

impl<K, V, const N: usize> Deref for KeyRef<'_, K, V, N> {
  type Target = K;

  fn deref(&self) -> &K {
    self.get()
  }
}

impl<K: fmt::Debug, V, const N: usize> fmt::Debug for KeyRef<'_, K, V, N> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.get().fmt(formatter)
  }
}

/// A read guard exposing the value and its associated key.
#[must_use = "the value's leaf remains read-locked while this guard is held"]
pub struct ValueRef<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Full record guard supplying the page latch and references.
  entry: EntryRef<'map, K, V, N>,
}

impl<K, V, const N: usize> ValueRef<'_, K, V, N> {
  /// Wraps a full entry guard without acquiring another latch.
  fn new(entry: EntryRef<'_, K, V, N>) -> ValueRef<'_, K, V, N> {
    ValueRef { entry }
  }

  /// Returns the key associated with this value.
  pub fn key(&self) -> &K {
    self.entry.key()
  }

  /// Returns the guarded value.
  pub fn get(&self) -> &V {
    self.entry.value()
  }
}

impl<K, V, const N: usize> Deref for ValueRef<'_, K, V, N> {
  type Target = V;

  fn deref(&self) -> &V {
    self.get()
  }
}

impl<K, V: fmt::Debug, const N: usize> fmt::Debug for ValueRef<'_, K, V, N> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.get().fmt(formatter)
  }
}

/// An ascending iterator over guarded map keys.
pub struct Keys<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Underlying full-record iterator.
  inner: Iter<'map, K, V, N>,
}

impl<'map, K, V, const N: usize> Keys<'map, K, V, N> {
  /// Creates a key adapter around a full iterator.
  pub(crate) fn new(map: &'map BLinkMap<K, V, N>) -> Self {
    Self {
      inner: Iter::new(map),
    }
  }
}

impl<'map, K: Ord, V, const N: usize> Iterator for Keys<'map, K, V, N> {
  type Item = KeyRef<'map, K, V, N>;

  fn next(&mut self) -> Option<Self::Item> {
    self.inner.next().map(KeyRef::new)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.inner.size_hint()
  }
}

impl<K: Ord, V, const N: usize> FusedIterator for Keys<'_, K, V, N> {}

/// An ascending iterator over guarded map values.
pub struct Values<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Underlying full-record iterator.
  inner: Iter<'map, K, V, N>,
}

impl<'map, K, V, const N: usize> Values<'map, K, V, N> {
  /// Creates a value adapter around a full iterator.
  pub(crate) fn new(map: &'map BLinkMap<K, V, N>) -> Self {
    Self {
      inner: Iter::new(map),
    }
  }
}

impl<'map, K: Ord, V, const N: usize> Iterator for Values<'map, K, V, N> {
  type Item = ValueRef<'map, K, V, N>;

  fn next(&mut self) -> Option<Self::Item> {
    self.inner.next().map(ValueRef::new)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.inner.size_hint()
  }
}

impl<K: Ord, V, const N: usize> FusedIterator for Values<'_, K, V, N> {}

/// Clones the shared key handle at a stable leaf index.
fn clone_key<K, V, const N: usize>(
  guard: &NodeReadGuard<K, V, N>,
  index: usize,
) -> KeyHandle<K> {
  match &guard.kind {
    NodeKind::Leaf(leaf) => leaf.entries[index].key.clone(),
    NodeKind::Internal(_) | NodeKind::Redirect(_) => {
      unreachable!("an iterator candidate must be a leaf record")
    },
  }
}

/// Enforces the same invalid-range panic conditions as standard ordered maps.
fn validate_bounds<Q: Ord + ?Sized, R: RangeBounds<Q>>(bounds: &R) {
  let (start, end) = match (bounds.start_bound(), bounds.end_bound()) {
    (Bound::Included(start), Bound::Included(end))
    | (Bound::Included(start), Bound::Excluded(end))
    | (Bound::Excluded(start), Bound::Included(end))
    | (Bound::Excluded(start), Bound::Excluded(end)) => (start, end),
    (Bound::Unbounded, _) | (_, Bound::Unbounded) => return,
  };

  assert!(start <= end, "range start is greater than range end");
  assert!(
    start != end
      || !matches!(bounds.start_bound(), Bound::Excluded(_))
      || !matches!(bounds.end_bound(), Bound::Excluded(_)),
    "range start and end are equal and excluded"
  );
}

/// Tests a candidate key against a range's upper bound.
fn inside_end_bound<K, Q, R>(key: &KeyHandle<K>, bounds: &R) -> bool
where
  K: Borrow<Q>,
  Q: Ord + ?Sized,
  R: RangeBounds<Q>,
{
  match bounds.end_bound() {
    Bound::Unbounded => true,
    Bound::Included(end) => key.get().borrow().cmp(end) != Ordering::Greater,
    Bound::Excluded(end) => key.get().borrow().cmp(end) == Ordering::Less,
  }
}

#[cfg(test)]
mod tests {
  use std::{
    ops::Bound,
    sync::{Arc, Barrier},
    thread,
  };

  use crate::BLinkMap;

  #[test]
  fn iteration_remains_sorted_after_splits_and_merges() {
    let map = BLinkMap::<u64, u64, 3>::new();
    for key in 0..1_000 {
      map.insert(key, key * 10);
    }
    for key in (0..1_000).step_by(3) {
      map.remove(&key);
    }

    let mut previous = None;
    let mut count = 0;
    for entry in &map {
      if let Some(previous) = previous {
        assert!(previous < *entry.key());
      }
      assert_eq!(*entry.value(), *entry.key() * 10);
      assert_ne!(*entry.key() % 3, 0);
      previous = Some(*entry.key());
      count += 1;
    }
    assert_eq!(count, map.len());
  }

  #[test]
  fn ranges_honor_owned_borrowed_and_unbounded_endpoints() {
    let map = BLinkMap::<String, usize, 4>::new();
    for (key, value) in [
      ("alpha", 1),
      ("bravo", 2),
      ("charlie", 3),
      ("delta", 4),
      ("echo", 5),
    ] {
      map.insert(String::from(key), value);
    }

    let mut range = map
      .range::<str, _>((Bound::Included("bravo"), Bound::Included("delta")));
    assert_eq!(range.next().unwrap().key(), "bravo");
    assert_eq!(range.next().unwrap().key(), "charlie");
    assert_eq!(range.next().unwrap().key(), "delta");
    assert!(range.next().is_none());

    let mut range =
      map.range::<str, _>((Bound::Excluded("bravo"), Bound::Unbounded));
    assert_eq!(range.next().unwrap().key(), "charlie");
  }

  #[test]
  #[should_panic(expected = "range start is greater")]
  fn inverted_range_panics() {
    let map = BLinkMap::<u64, (), 3>::new();
    let start = 9;
    let end = 3;
    let _ = map.range(start..end);
  }

  #[test]
  fn concurrent_iterator_is_monotonic_while_pages_change() {
    let map = Arc::new(BLinkMap::<u64, u64, 5>::new());
    for key in 0..2_000 {
      map.insert(key * 2, key);
    }

    let start = Arc::new(Barrier::new(2));
    let writer_map = Arc::clone(&map);
    let writer_start = Arc::clone(&start);
    let writer = thread::spawn(move || {
      writer_start.wait();
      for key in 0..2_000 {
        writer_map.insert(key * 2 + 1, key);
      }
      for key in (0..4_000).step_by(4) {
        writer_map.remove(&key);
      }
    });

    start.wait();
    let mut previous = None;
    for entry in map.iter() {
      if let Some(previous) = previous {
        assert!(previous < *entry.key());
      }
      previous = Some(*entry.key());
    }

    writer.join().unwrap();
    map.assert_valid();
  }
}
