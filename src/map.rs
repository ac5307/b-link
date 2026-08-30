use std::{borrow::Borrow, fmt, iter::FromIterator, ops::RangeBounds};

use crate::{
  handle::{EntryMut, EntryRef},
  iter::{Iter, Keys, Range, Values},
  node::Tree,
};

/// The maximum stable entries/separators per page when [`BLinkMap`] is named
/// without an explicit const capacity argument.
pub const DEFAULT_PAGE_CAPACITY: usize = 32;

/// A concurrent ordered map backed by a B-link tree.
///
/// `N` is the exact allocation capacity for separators or entries in a page.
/// Full pages split before another item is inserted. `N` must be at least
/// three; omitting it uses [`DEFAULT_PAGE_CAPACITY`].
///
/// Reads and ordinary writes use one page latch. Splits, rotations, and merges
/// use top-down latch crabbing: a writer briefly owns a parent and one or two
/// adjacent children, repairs them, then releases the parent before descending.
/// No tree-wide structure lock serializes writers. Each page carries lower and
/// upper fences, a right sibling, and a forwarding state for stale references,
/// so completed key operations are linearizable at their destination leaf.
///
/// # Key ordering
///
/// Keys must retain the same ordering for their entire time in the map. Do not
/// mutate ordering-relevant state through interior mutability. For borrowed
/// lookups, the ordering of `K: Borrow<Q>` must agree with both `K::Ord` and
/// `Q::Ord`, just as it must for standard ordered maps.
///
/// # Entry guards and locking
///
/// [`EntryRef`] and [`EntryMut`] lock an entire leaf, not only one record. Drop
/// a guard before invoking another operation that could target that leaf.
/// Re-entering the map while a conflicting guard is alive can deadlock, and
/// acquiring several guards in inconsistent orders can create lock cycles.
pub struct BLinkMap<K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Internal page owner and algorithm implementation.
  pub(crate) tree: Tree<K, V, N>,
}

impl<K, V, const N: usize> BLinkMap<K, V, N> {
  /// Creates an empty map with one allocated leaf page.
  ///
  /// # Panics
  ///
  /// Panics when `N < 3`, when the capacity arithmetic overflows, or when a
  /// backing page layout cannot be represented. Allocation failure is handled
  /// by Rust's global allocation-error handler.
  pub fn new() -> Self {
    Self { tree: Tree::new() }
  }

  /// Returns the number of entries currently in the map.
  ///
  /// During concurrent mutation this is a momentary atomic observation, not a
  /// transactionally consistent snapshot of other operations.
  pub fn len(&self) -> usize {
    self.tree.len()
  }

  /// Returns true when the map has no entries at the instant it is observed.
  ///
  /// Like [`BLinkMap::len`], this result can become stale immediately when
  /// other threads mutate the map.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Removes every entry and returns the map to one empty root leaf.
  ///
  /// The exclusive `&mut self` borrow proves that no entry guard or concurrent
  /// operation can still reference the old tree through this map. Individual
  /// pages are reclaimed as their internal `Arc` owners are dropped.
  pub fn clear(&mut self) {
    self.tree = Tree::new();
  }
}

impl<K: Ord, V, const N: usize> BLinkMap<K, V, N> {
  /// Returns a read guard for `key`, or `None` when the key is absent.
  ///
  /// The returned guard keeps the whole destination leaf read-locked. Its key
  /// and value references remain stable, but it must be dropped before a
  /// conflicting operation on that leaf to avoid deadlock.
  ///
  /// The borrowed form `Q` must use an ordering consistent with `K`.
  pub fn get<Q>(&self, key: &Q) -> Option<EntryRef<'_, K, V, N>>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (guard, index) = self.tree.get(key)?;
    Some(EntryRef::new(guard, index))
  }

  /// Returns a read guard exposing both the stored key and value.
  ///
  /// This is an explicitly named counterpart to [`BLinkMap::get`]; both
  /// return the same [`EntryRef`] because that guard already supplies
  /// [`EntryRef::key`] and [`EntryRef::value`].
  pub fn get_key_value<Q>(&self, key: &Q) -> Option<EntryRef<'_, K, V, N>>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.get(key)
  }

  /// Returns a write guard for `key`, or `None` when the key is absent.
  ///
  /// The guard exclusively locks the entire destination leaf and supports
  /// mutation through [`EntryMut::value_mut`] or `DerefMut`. Drop it before
  /// calling any other map operation that could visit the same leaf.
  ///
  /// The borrowed form `Q` must use an ordering consistent with `K`.
  pub fn get_mut<Q>(&self, key: &Q) -> Option<EntryMut<'_, K, V, N>>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (guard, index) = self.tree.get_mut(key)?;
    Some(EntryMut::new(guard, index))
  }

  /// Returns true when the map contains `key` at the operation's linearization
  /// point. The temporary leaf read latch is released before this returns.
  ///
  /// The borrowed form `Q` must use an ordering consistent with `K`.
  pub fn contains_key<Q>(&self, key: &Q) -> bool
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.get(key).is_some()
  }

  /// Inserts `key` and `value`, returning the previous value if the key was
  /// already present.
  ///
  /// A non-splitting insert locks only its destination leaf. A structural
  /// insert splits full pages top-down while holding only the affected parent
  /// and child, allowing writers in disjoint subtrees to continue concurrently.
  pub fn insert(&self, key: K, value: V) -> Option<V> {
    self.tree.insert(key, value)
  }

  /// Removes `key`, returning its value when present.
  ///
  /// Before descending into a minimally occupied page, deletion borrows from a
  /// sibling or merges two pages. Merged pages are removed from their parent
  /// and sibling chain, while stale `Arc` handles see a forwarding page until
  /// they drain. Empty internal root levels are collapsed before this returns.
  /// A removed key that also serves as a surviving fence or separator may keep
  /// its shared key allocation alive until that boundary changes or the map is
  /// dropped; the user value is returned immediately.
  ///
  /// The borrowed form `Q` must use an ordering consistent with `K`.
  pub fn remove<Q>(&self, key: &Q) -> Option<V>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self.tree.remove(key)
  }

  /// Returns the least key/value record, guarded by its leaf read latch.
  pub fn first_key_value(&self) -> Option<EntryRef<'_, K, V, N>> {
    let (guard, index) = self.tree.first_entry()?;
    Some(EntryRef::new(guard, index))
  }

  /// Returns the greatest key/value record, guarded by its leaf read latch.
  pub fn last_key_value(&self) -> Option<EntryRef<'_, K, V, N>> {
    let (guard, index) = self.tree.last_entry()?;
    Some(EntryRef::new(guard, index))
  }

  /// Returns a weakly consistent ascending iterator over guarded records.
  ///
  /// See [`Iter`] for its concurrent visibility and leaf-locking semantics.
  pub fn iter(&self) -> Iter<'_, K, V, N> {
    Iter::new(self)
  }

  /// Returns a weakly consistent ascending iterator over guarded keys.
  pub fn keys(&self) -> Keys<'_, K, V, N> {
    Keys::new(self)
  }

  /// Returns a weakly consistent ascending iterator over guarded values.
  pub fn values(&self) -> Values<'_, K, V, N> {
    Values::new(self)
  }

  /// Returns an ascending iterator over records whose keys lie in `bounds`.
  ///
  /// Borrowed bounds are supported when their ordering agrees with `K`.
  ///
  /// # Panics
  ///
  /// Panics when the start is greater than the end, or when equal start and
  /// end bounds are both excluded.
  pub fn range<Q, R>(&self, bounds: R) -> Range<'_, K, V, Q, R, N>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
    R: RangeBounds<Q>,
  {
    Range::new(self, bounds)
  }

  #[cfg(test)]
  pub(crate) fn assert_valid(&self) {
    if let Err(error) = self.tree.validate() {
      panic!("B-link tree invariant violated: {error}");
    }
  }

  #[cfg(test)]
  fn height(&self) -> usize {
    self.tree.height()
  }
}

impl<K, V, const N: usize> Default for BLinkMap<K, V, N> {
  fn default() -> Self {
    Self::new()
  }
}

impl<K: Ord, V, const N: usize> Extend<(K, V)> for BLinkMap<K, V, N> {
  /// Inserts every pair, replacing values for duplicate keys.
  fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iterable: T) {
    for (key, value) in iterable {
      self.insert(key, value);
    }
  }
}

impl<K: Ord, V, const N: usize> FromIterator<(K, V)> for BLinkMap<K, V, N> {
  /// Builds a map by repeated concurrent-map insertions.
  fn from_iter<T: IntoIterator<Item = (K, V)>>(iterable: T) -> Self {
    let mut map = Self::new();
    map.extend(iterable);
    map
  }
}

impl<'map, K: Ord, V, const N: usize> IntoIterator
  for &'map BLinkMap<K, V, N>
{
  type Item = EntryRef<'map, K, V, N>;
  type IntoIter = Iter<'map, K, V, N>;

  /// Creates the same guarded iterator as [`BLinkMap::iter`].
  fn into_iter(self) -> Self::IntoIter {
    self.iter()
  }
}

impl<K: Ord + fmt::Debug, V: fmt::Debug, const N: usize> fmt::Debug
  for BLinkMap<K, V, N>
{
  /// Formats a weakly consistent ordered view, dropping each guard promptly.
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    let mut debug = formatter.debug_map();
    let mut iterator = self.iter();

    for entry in &mut iterator {
      debug.entry(entry.key(), entry.value());
    }

    debug.finish()
  }
}

#[cfg(test)]
mod tests {
  use std::{
    collections::BTreeMap,
    sync::{mpsc, Arc, Barrier},
    thread,
    time::Duration,
  };

  use super::*;

  type TestMap = BLinkMap<u64, u64, 3>;

  #[test]
  fn basic_map_operations_and_entry_guards() {
    let map = TestMap::new();

    assert!(map.is_empty());
    assert_eq!(map.insert(10, 100), None);
    assert_eq!(map.insert(20, 200), None);
    assert_eq!(map.insert(10, 500), Some(100));
    assert_eq!(map.len(), 2);
    assert!(map.contains_key(&20));

    let entry = map.get(&10).unwrap();
    assert_eq!(entry.key(), &10);
    assert_eq!(entry.value(), &500);
    drop(entry);

    {
      let mut entry = map.get_mut(&10).unwrap();
      assert_eq!(entry.key(), &10);
      *entry += 1;
    }

    assert_eq!(map.remove(&10), Some(501));
    assert_eq!(map.remove(&10), None);
    assert_eq!(map.len(), 1);
    map.assert_valid();
  }

  #[test]
  fn grows_through_leaf_internal_and_root_splits() {
    let map = TestMap::new();

    for key in 0..2_000 {
      assert_eq!(map.insert(key, key * 10), None);
    }

    assert!(map.height() >= 4);
    map.assert_valid();

    for key in 0..2_000 {
      assert_eq!(map.get(&key).map(|entry| *entry), Some(key * 10));
    }
  }

  #[test]
  fn reverse_insertion_complete_deletion_and_reinsertion() {
    let map = TestMap::new();

    for key in (0..1_000).rev() {
      map.insert(key, key);
    }
    for key in 0..1_000 {
      assert_eq!(map.remove(&key), Some(key));
    }

    assert!(map.is_empty());
    map.assert_valid();

    for key in (0..1_000).rev() {
      assert_eq!(map.insert(key, key + 1), None);
    }

    map.assert_valid();
  }

  fn differential_trace<const CAPACITY: usize>() {
    let map = BLinkMap::<u64, u64, CAPACITY>::new();
    let mut reference = BTreeMap::new();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;

    for step in 0..20_000 {
      state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);

      let key = (state >> 17) % 400;
      let value = state.rotate_left(23);

      match state % 4 {
        0 | 1 => {
          assert_eq!(map.insert(key, value), reference.insert(key, value));
        },
        2 => {
          assert_eq!(map.remove(&key), reference.remove(&key));
        },
        _ => {
          assert_eq!(
            map.get(&key).map(|entry| *entry),
            reference.get(&key).copied()
          );
        },
      }

      if step % 257 == 0 {
        map.assert_valid();
      }
    }

    assert_eq!(map.len(), reference.len());
    for key in 0..400 {
      assert_eq!(
        map.get(&key).map(|entry| *entry),
        reference.get(&key).copied()
      );
    }
    map.assert_valid();
  }

  #[test]
  fn matches_btree_map_at_multiple_page_capacities() {
    differential_trace::<3>();
    differential_trace::<4>();
    differential_trace::<5>();
    differential_trace::<8>();
  }

  #[test]
  fn borrowed_lookup_does_not_require_cloning_keys() {
    #[derive(Eq, Ord, PartialEq, PartialOrd)]
    struct NonClone(u64);

    let map = BLinkMap::<NonClone, u64, 3>::new();
    map.insert(NonClone(7), 70);

    assert_eq!(map.get(&NonClone(7)).map(|entry| *entry), Some(70));

    let strings = BLinkMap::<String, u64, 3>::new();
    strings.insert(String::from("alpha"), 1);
    assert!(strings.contains_key("alpha"));
    assert_eq!(strings.get("alpha").map(|entry| *entry), Some(1));
    *strings.get_mut("alpha").unwrap() = 2;
    assert_eq!(strings.remove("alpha"), Some(2));
    assert!(!strings.contains_key("alpha"));
  }

  #[test]
  #[should_panic(expected = "capacity must be at least 3")]
  fn rejects_too_small_page_capacity() {
    let _ = BLinkMap::<u64, u64, 2>::new();
  }

  #[test]
  fn concurrent_disjoint_inserts() {
    let map = Arc::new(BLinkMap::<u64, u64, 8>::new());

    let threads = std::array::from_fn::<_, 8, _>(|thread_id| {
      let map = Arc::clone(&map);

      thread::spawn(move || {
        for offset in 0..1_000 {
          let key = thread_id as u64 * 1_000 + offset;
          assert_eq!(map.insert(key, key), None);
        }
      })
    });

    for thread in threads {
      thread.join().unwrap();
    }

    assert_eq!(map.len(), 8_000);
    map.assert_valid();

    for key in 0..8_000 {
      assert_eq!(map.get(&key).map(|entry| *entry), Some(key));
    }
  }

  #[test]
  fn concurrent_same_key_operations_are_linearizable() {
    const THREADS: usize = 8;

    let map = Arc::new(BLinkMap::<u64, u64, 3>::new());
    let barrier = Arc::new(Barrier::new(THREADS));
    let inserters = std::array::from_fn::<_, THREADS, _>(|value| {
      let map = Arc::clone(&map);
      let barrier = Arc::clone(&barrier);

      thread::spawn(move || {
        barrier.wait();
        map.insert(1, value as u64)
      })
    });

    // One insert creates the entry. Every later insert returns the value it
    // replaced, while exactly one submitted value remains in the leaf.
    let mut saw_value = [false; THREADS];
    let mut new_key_results = 0;

    for inserter in inserters {
      match inserter.join().unwrap() {
        Some(value) => {
          let slot = &mut saw_value[value as usize];
          assert!(!*slot);
          *slot = true;
        },
        None => new_key_results += 1,
      }
    }

    assert_eq!(new_key_results, 1);
    let final_value = *map.get(&1).unwrap();
    assert!(!saw_value[final_value as usize]);
    saw_value[final_value as usize] = true;
    assert!(saw_value.into_iter().all(|seen| seen));
    assert_eq!(map.len(), 1);

    let barrier = Arc::new(Barrier::new(THREADS));
    let removers = std::array::from_fn::<_, THREADS, _>(|_| {
      let map = Arc::clone(&map);
      let barrier = Arc::clone(&barrier);

      thread::spawn(move || {
        barrier.wait();
        map.remove(&1)
      })
    });

    let mut removed = 0;
    for remover in removers {
      if remover.join().unwrap().is_some() {
        removed += 1;
      }
    }

    assert_eq!(removed, 1);
    assert!(map.is_empty());
    map.assert_valid();
  }

  #[test]
  fn stable_keys_never_disappear_during_splits() {
    const READERS: usize = 4;

    let map = Arc::new(BLinkMap::<u64, u64, 3>::new());
    for key in 0..1_000 {
      map.insert(key * 2, key);
    }

    let barrier = Arc::new(Barrier::new(READERS + 1));
    let writer_map = Arc::clone(&map);
    let writer_barrier = Arc::clone(&barrier);
    let writer = thread::spawn(move || {
      writer_barrier.wait();
      for key in 0..1_000 {
        writer_map.insert(key * 2 + 1, key);
      }
    });

    let readers = std::array::from_fn::<_, READERS, _>(|_| {
      let map = Arc::clone(&map);
      let barrier = Arc::clone(&barrier);

      thread::spawn(move || {
        barrier.wait();
        for _ in 0..4 {
          for key in 0..1_000 {
            assert_eq!(map.get(&(key * 2)).map(|entry| *entry), Some(key));
          }
        }
      })
    });

    writer.join().unwrap();
    for reader in readers {
      reader.join().unwrap();
    }

    map.assert_valid();
  }

  #[test]
  fn stable_keys_never_disappear_during_rotations_and_merges() {
    const READERS: usize = 4;

    let map = Arc::new(BLinkMap::<u64, u64, 5>::new());
    for key in 0..4_000 {
      map.insert(key, key * 10);
    }

    let barrier = Arc::new(Barrier::new(READERS + 1));
    let remover_map = Arc::clone(&map);
    let remover_barrier = Arc::clone(&barrier);
    let remover = thread::spawn(move || {
      remover_barrier.wait();
      for key in (1..4_000).step_by(2) {
        assert_eq!(remover_map.remove(&key), Some(key * 10));
      }
    });

    let readers = std::array::from_fn::<_, READERS, _>(|_| {
      let map = Arc::clone(&map);
      let barrier = Arc::clone(&barrier);
      thread::spawn(move || {
        barrier.wait();
        for _ in 0..3 {
          for key in (0..4_000).step_by(2) {
            assert_eq!(map.get(&key).map(|entry| *entry), Some(key * 10));
          }
        }
      })
    });

    remover.join().unwrap();
    for reader in readers {
      reader.join().unwrap();
    }

    assert_eq!(map.len(), 2_000);
    map.assert_valid();
  }

  #[test]
  fn concurrent_get_mut_and_remove() {
    let map = Arc::new(BLinkMap::<u64, u64, 8>::new());
    for key in 0..4_000 {
      map.insert(key, 0);
    }

    let writers = std::array::from_fn::<_, 4, _>(|thread_id| {
      let map = Arc::clone(&map);

      thread::spawn(move || {
        for offset in 0..1_000 {
          let key = thread_id as u64 * 1_000 + offset;
          *map.get_mut(&key).unwrap() += 1;
        }
      })
    });

    for writer in writers {
      writer.join().unwrap();
    }

    let removers = std::array::from_fn::<_, 4, _>(|thread_id| {
      let map = Arc::clone(&map);

      thread::spawn(move || {
        for offset in 0..500 {
          let key = thread_id as u64 * 1_000 + offset;
          assert_eq!(map.remove(&key), Some(1));
        }
      })
    });

    for remover in removers {
      remover.join().unwrap();
    }

    assert_eq!(map.len(), 2_000);
    map.assert_valid();
  }

  #[test]
  fn held_leaf_guard_does_not_convoy_a_distant_structural_writer() {
    let map = Arc::new(BLinkMap::<u64, u64, 5>::new());
    for key in 0..2_000 {
      map.insert(key, key);
    }

    // Retain the far-left leaf latch while another thread grows and repeatedly
    // splits the far-right branch. Ancestor crabbing uses try-latches, so it
    // never waits for this leaf while holding the root or a shared ancestor.
    let held_left_leaf = map.get_mut(&0).unwrap();
    let worker_map = Arc::clone(&map);
    let (finished_tx, finished_rx) = mpsc::channel();
    let worker = thread::spawn(move || {
      for key in 2_000..4_000 {
        worker_map.insert(key, key);
      }
      finished_tx.send(()).unwrap();
    });

    finished_rx.recv_timeout(Duration::from_secs(5)).expect(
      "a distant structural writer was convoyed behind a leaf guard",
    );
    drop(held_left_leaf);
    worker.join().unwrap();

    assert_eq!(map.len(), 4_000);
    map.assert_valid();
  }
}
