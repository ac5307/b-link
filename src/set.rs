use std::{
  borrow::Borrow,
  fmt,
  iter::{FromIterator, FusedIterator},
  ops::{Deref, RangeBounds},
};

use crate::{
  iter::{Iter, Range},
  map::{BLinkMap, DEFAULT_PAGE_CAPACITY},
  EntryRef,
};

/// A concurrent ordered set backed by [`BLinkMap`].
///
/// Values are stored as map keys with zero-sized `()` payloads, so the set has
/// the same page-latch concurrency, fence correction, split, merge, and memory
/// reclamation behavior as the map. `N` is the maximum stable records or
/// separators per page and defaults to [`DEFAULT_PAGE_CAPACITY`].
///
/// Lookup and iteration return [`SetRef`] guards rather than bare references.
/// Each guard read-locks its whole leaf; drop it before a conflicting operation
/// on the same part of the set.
pub struct BLinkSet<T, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Map storage; unit values add no per-record payload bytes.
  map: BLinkMap<T, (), N>,
}

impl<T, const N: usize> BLinkSet<T, N> {
  /// Creates an empty set.
  ///
  /// # Panics
  ///
  /// Panics under the same invalid-capacity and allocation conditions as
  /// [`BLinkMap::new`].
  pub fn new() -> Self {
    Self {
      map: BLinkMap::new(),
    }
  }

  /// Returns the number of values observed by the set's atomic counter.
  pub fn len(&self) -> usize {
    self.map.len()
  }

  /// Returns true when the set is empty at the instant it is observed.
  pub fn is_empty(&self) -> bool {
    self.map.is_empty()
  }

  /// Removes every value and restores one empty root leaf.
  pub fn clear(&mut self) {
    self.map.clear();
  }
}

impl<T: Ord, const N: usize> BLinkSet<T, N> {
  /// Returns true when an ordering-compatible borrowed value is present.
  pub fn contains<Q>(&self, value: &Q) -> bool
  where
    T: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.map.contains_key(value)
  }

  /// Returns a guard for the stored value equal to `value`.
  pub fn get<Q>(&self, value: &Q) -> Option<SetRef<'_, T, N>>
  where
    T: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.map.get(value).map(SetRef::new)
  }

  /// Inserts `value`, returning true only when it was not already present.
  pub fn insert(&self, value: T) -> bool {
    self.map.insert(value, ()).is_none()
  }

  /// Removes `value`, returning true only when a record was present.
  ///
  /// Deletion performs the map's complete borrow/merge/unlink/root-collapse
  /// protocol; it does not leave empty non-root leaf pages behind. A removed
  /// value that remains useful as a routing fence may keep its allocation alive
  /// until that fence changes or the set is dropped.
  pub fn remove<Q>(&self, value: &Q) -> bool
  where
    T: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.map.remove(value).is_some()
  }

  /// Returns a guard for the least value.
  pub fn first(&self) -> Option<SetRef<'_, T, N>> {
    self.map.first_key_value().map(SetRef::new)
  }

  /// Returns a guard for the greatest value.
  pub fn last(&self) -> Option<SetRef<'_, T, N>> {
    self.map.last_key_value().map(SetRef::new)
  }

  /// Returns a weakly consistent ascending iterator over guarded values.
  pub fn iter(&self) -> SetIter<'_, T, N> {
    SetIter::new(self)
  }

  /// Returns a weakly consistent ascending iterator over `bounds`.
  ///
  /// # Panics
  ///
  /// Panics for an inverted range or equal bounds that are both excluded.
  pub fn range<Q, R>(&self, bounds: R) -> SetRange<'_, T, Q, R, N>
  where
    T: Borrow<Q>,
    Q: Ord + ?Sized,
    R: RangeBounds<Q>,
  {
    SetRange {
      inner: self.map.range(bounds),
    }
  }

  /// Returns true when this set and `other` share no value.
  ///
  /// Concurrent mutation makes this a weakly consistent traversal rather than
  /// a transaction spanning both sets.
  pub fn is_disjoint(&self, other: &Self) -> bool {
    if std::ptr::eq(self, other) {
      return self.is_empty();
    }

    let (smaller, larger) = if self.len() <= other.len() {
      (self, other)
    } else {
      (other, self)
    };

    for value in smaller {
      if larger.contains(value.get()) {
        return false;
      }
    }
    true
  }

  /// Returns true when every observed value in this set is in `other`.
  ///
  /// Concurrent mutation makes this a weakly consistent traversal rather than
  /// an atomic relation between two sets.
  pub fn is_subset(&self, other: &Self) -> bool {
    if std::ptr::eq(self, other) {
      return true;
    }
    if self.len() > other.len() {
      return false;
    }

    for value in self {
      if !other.contains(value.get()) {
        return false;
      }
    }
    true
  }

  /// Returns true when every observed value in `other` is in this set.
  pub fn is_superset(&self, other: &Self) -> bool {
    other.is_subset(self)
  }
}

impl<T, const N: usize> Default for BLinkSet<T, N> {
  fn default() -> Self {
    Self::new()
  }
}

impl<T: Ord, const N: usize> Extend<T> for BLinkSet<T, N> {
  /// Inserts every value, ignoring duplicates.
  fn extend<I: IntoIterator<Item = T>>(&mut self, iterable: I) {
    for value in iterable {
      self.insert(value);
    }
  }
}

impl<T: Ord, const N: usize> FromIterator<T> for BLinkSet<T, N> {
  /// Builds a set from an arbitrary value iterator.
  fn from_iter<I: IntoIterator<Item = T>>(iterable: I) -> Self {
    let mut set = Self::new();
    set.extend(iterable);
    set
  }
}

impl<T: Ord, const N: usize> PartialEq for BLinkSet<T, N> {
  /// Compares weakly consistent ordered views of both sets.
  fn eq(&self, other: &Self) -> bool {
    std::ptr::eq(self, other)
      || (self.len() == other.len() && self.is_subset(other))
  }
}

impl<T: Ord, const N: usize> Eq for BLinkSet<T, N> {}

impl<T: Ord + fmt::Debug, const N: usize> fmt::Debug for BLinkSet<T, N> {
  /// Formats a weakly consistent ascending view of the set.
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut debug = formatter.debug_set();
    for value in self {
      debug.entry(value.get());
    }
    debug.finish()
  }
}

/// A read guard for one value stored in a [`BLinkSet`].
///
/// The guard owns the underlying map entry guard and therefore read-locks the
/// entire leaf until it is dropped.
#[must_use = "the set value's leaf remains read-locked while held"]
pub struct SetRef<'set, T, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Unit-valued map record supplying the key reference and latch.
  entry: EntryRef<'set, T, (), N>,
}

impl<T, const N: usize> SetRef<'_, T, N> {
  /// Wraps the backing map's entry guard.
  fn new(entry: EntryRef<'_, T, (), N>) -> SetRef<'_, T, N> {
    SetRef { entry }
  }

  /// Returns the guarded set value.
  pub fn get(&self) -> &T {
    self.entry.key()
  }
}

impl<T, const N: usize> Deref for SetRef<'_, T, N> {
  type Target = T;

  fn deref(&self) -> &T {
    self.get()
  }
}

impl<T: fmt::Debug, const N: usize> fmt::Debug for SetRef<'_, T, N> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.get().fmt(formatter)
  }
}

/// A weakly consistent ascending iterator over a [`BLinkSet`].
pub struct SetIter<'set, T, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Backing map record iterator.
  inner: Iter<'set, T, (), N>,
}

impl<'set, T: Ord, const N: usize> SetIter<'set, T, N> {
  /// Creates a set adapter over the map iterator.
  fn new(set: &'set BLinkSet<T, N>) -> Self {
    Self {
      inner: set.map.iter(),
    }
  }
}

impl<'set, T: Ord, const N: usize> Iterator for SetIter<'set, T, N> {
  type Item = SetRef<'set, T, N>;

  fn next(&mut self) -> Option<Self::Item> {
    self.inner.next().map(SetRef::new)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.inner.size_hint()
  }
}

impl<T: Ord, const N: usize> FusedIterator for SetIter<'_, T, N> {}

/// A weakly consistent ascending iterator over a bounded set range.
pub struct SetRange<
  'set,
  T,
  Q: ?Sized,
  R,
  const N: usize = DEFAULT_PAGE_CAPACITY,
> {
  /// Backing map range iterator.
  inner: Range<'set, T, (), Q, R, N>,
}

impl<'set, T, Q, R, const N: usize> Iterator for SetRange<'set, T, Q, R, N>
where
  T: Ord + Borrow<Q>,
  Q: Ord + ?Sized,
  R: RangeBounds<Q>,
{
  type Item = SetRef<'set, T, N>;

  fn next(&mut self) -> Option<Self::Item> {
    self.inner.next().map(SetRef::new)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    self.inner.size_hint()
  }
}

impl<T, Q, R, const N: usize> FusedIterator for SetRange<'_, T, Q, R, N>
where
  T: Ord + Borrow<Q>,
  Q: Ord + ?Sized,
  R: RangeBounds<Q>,
{
}

impl<'set, T: Ord, const N: usize> IntoIterator for &'set BLinkSet<T, N> {
  type Item = SetRef<'set, T, N>;
  type IntoIter = SetIter<'set, T, N>;

  /// Creates the same guarded iterator as [`BLinkSet::iter`].
  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

#[cfg(test)]
mod tests {
  use std::{
    sync::{Arc, Barrier},
    thread,
  };

  use super::*;

  #[test]
  fn set_operations_ranges_and_relations() {
    let set = BLinkSet::<u64, 3>::new();
    for value in 0..100 {
      assert!(set.insert(value));
      assert!(!set.insert(value));
    }

    assert_eq!(set.first().map(|value| *value), Some(0));
    assert_eq!(set.last().map(|value| *value), Some(99));

    let mut expected = 20;
    for value in set.range(20..30) {
      assert_eq!(*value, expected);
      expected += 1;
    }
    assert_eq!(expected, 30);

    let evens = BLinkSet::<u64, 3>::new();
    for value in (0..100).step_by(2) {
      evens.insert(value);
    }
    assert!(evens.is_subset(&set));
    assert!(set.is_superset(&evens));
    assert!(!set.is_disjoint(&evens));

    let distant = BLinkSet::<u64, 3>::new();
    distant.insert(1_000);
    assert!(set.is_disjoint(&distant));

    for value in 0..100 {
      assert!(set.remove(&value));
    }
    assert!(set.is_empty());
    set.map.assert_valid();
  }

  #[test]
  fn set_collect_extend_and_clear() {
    let mut set: BLinkSet<u64, 4> = (0..50).collect();
    set.extend(25..75);
    assert_eq!(set.len(), 75);
    assert!(format!("{set:?}").starts_with("{0, 1, 2"));
    set.clear();
    assert!(set.is_empty());
  }

  #[test]
  fn concurrent_set_writers_split_and_merge() {
    const THREADS: usize = 6;
    const PER_THREAD: usize = 500;

    let set = Arc::new(BLinkSet::<usize, 5>::new());
    let start = Arc::new(Barrier::new(THREADS));
    let workers = std::array::from_fn::<_, THREADS, _>(|thread_index| {
      let set = Arc::clone(&set);
      let start = Arc::clone(&start);
      thread::spawn(move || {
        start.wait();
        for offset in 0..PER_THREAD {
          assert!(set.insert(thread_index * PER_THREAD + offset));
        }
        for offset in (0..PER_THREAD).step_by(2) {
          assert!(set.remove(&(thread_index * PER_THREAD + offset)));
        }
      })
    });

    for worker in workers {
      worker.join().unwrap();
    }

    assert_eq!(set.len(), THREADS * PER_THREAD / 2);
    set.map.assert_valid();
  }
}
