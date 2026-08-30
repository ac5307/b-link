use std::{
  cmp::Ordering,
  fmt,
  marker::PhantomData,
  ops::{Deref, DerefMut},
  sync::Arc,
};

use crate::{
  map::{BLinkMap, DEFAULT_PAGE_CAPACITY},
  node::{NodeKind, NodeReadGuard, NodeWriteGuard},
};

#[repr(transparent)]
/// Reference-counted ownership of a stored key.
///
/// Leaf entries, internal separators, and high fences can share the same key
/// allocation. This avoids imposing `K: Clone` on the public map API.
pub(crate) struct KeyHandle<K> {
  /// The one owned key allocation shared by tree metadata.
  inner: Arc<K>,
}

impl<K> KeyHandle<K> {
  /// Allocates a new shared handle around an inserted key.
  pub(crate) fn new(key: K) -> Self {
    Self {
      inner: Arc::new(key),
    }
  }

  #[inline]
  /// Borrows the underlying key.
  pub(crate) fn get(&self) -> &K {
    &self.inner
  }
}

impl<K> AsRef<K> for KeyHandle<K> {
  fn as_ref(&self) -> &K {
    self.get()
  }
}

impl<K> Deref for KeyHandle<K> {
  type Target = K;

  fn deref(&self) -> &K {
    self.get()
  }
}

impl<K> Clone for KeyHandle<K> {
  fn clone(&self) -> Self {
    Self {
      inner: Arc::clone(&self.inner),
    }
  }
}

impl<K: PartialEq> PartialEq for KeyHandle<K> {
  fn eq(&self, other: &Self) -> bool {
    self.inner.eq(&other.inner)
  }
}

impl<K: Eq> Eq for KeyHandle<K> {}

impl<K: PartialOrd> PartialOrd for KeyHandle<K> {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    self.inner.partial_cmp(&other.inner)
  }
}

impl<K: Ord> Ord for KeyHandle<K> {
  fn cmp(&self, other: &Self) -> Ordering {
    self.inner.cmp(&other.inner)
  }
}

impl<K: std::hash::Hash> std::hash::Hash for KeyHandle<K> {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.inner.hash(state);
  }
}

/// A read guard for an entry in a [`BLinkMap`].
///
/// The guard keeps its leaf read-locked. This makes the returned key and value
/// references stable until the guard is dropped.
///
/// # Locking
///
/// Drop this guard before calling a map operation that may need the same leaf,
/// especially [`BLinkMap::insert`], [`BLinkMap::remove`], or
/// [`BLinkMap::get_mut`]. Retaining it can self-deadlock. Holding several entry
/// guards while acquiring more guards in inconsistent orders can likewise
/// create a cross-thread lock cycle.
#[must_use = "the entry remains read-locked only while this guard is held"]
pub struct EntryRef<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Owning page latch that keeps the leaf and its contents stable.
  guard: NodeReadGuard<K, V, N>,
  /// Entry location, stable because writers cannot acquire the leaf latch.
  index: usize,
  /// Connects the owning guard's lifetime to the map borrow in the public API.
  _map: PhantomData<&'map BLinkMap<K, V, N>>,
}

impl<K, V, const N: usize> EntryRef<'_, K, V, N> {
  /// Wraps a leaf latch and the matching index found under that latch.
  pub(crate) fn new(guard: NodeReadGuard<K, V, N>, index: usize) -> Self {
    Self {
      guard,
      index,
      _map: PhantomData,
    }
  }

  /// Returns the entry's immutable key.
  pub fn key(&self) -> &K {
    &self.entry().key
  }

  /// Returns the entry's immutable value.
  pub fn value(&self) -> &V {
    &self.entry().value
  }

  /// Resolves the stable index inside the read-latched leaf.
  fn entry(&self) -> &crate::node::Entry<K, V> {
    match &self.guard.kind {
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("an entry guard must hold a leaf")
      },
      NodeKind::Leaf(leaf) => &leaf.entries[self.index],
    }
  }
}

impl<K, V, const N: usize> Deref for EntryRef<'_, K, V, N> {
  type Target = V;

  fn deref(&self) -> &V {
    self.value()
  }
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize> fmt::Debug
  for EntryRef<'_, K, V, N>
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("EntryRef")
      .field("key", self.key())
      .field("value", self.value())
      .finish()
  }
}

/// A write guard for an entry in a [`BLinkMap`].
///
/// The guard keeps its leaf write-locked. Mutations through [`DerefMut`] are
/// visible to later map operations when the guard is dropped.
///
/// # Locking
///
/// While this guard exists, every other operation targeting the same leaf is
/// blocked—including reads. Drop it before calling back into the map. Acquiring
/// multiple mutable guards in different orders across threads can deadlock.
#[must_use = "the entry remains write-locked only while this guard is held"]
pub struct EntryMut<'map, K, V, const N: usize = DEFAULT_PAGE_CAPACITY> {
  /// Owning exclusive page latch.
  guard: NodeWriteGuard<K, V, N>,
  /// Entry location, stable for the duration of the exclusive latch.
  index: usize,
  /// Connects the owning guard's lifetime to the map borrow in the public API.
  _map: PhantomData<&'map BLinkMap<K, V, N>>,
}

impl<K, V, const N: usize> EntryMut<'_, K, V, N> {
  /// Wraps an exclusive leaf latch and the matching index found under it.
  pub(crate) fn new(guard: NodeWriteGuard<K, V, N>, index: usize) -> Self {
    Self {
      guard,
      index,
      _map: PhantomData,
    }
  }

  /// Returns the entry's immutable key.
  pub fn key(&self) -> &K {
    &self.entry().key
  }

  /// Returns the entry's immutable value.
  pub fn value(&self) -> &V {
    &self.entry().value
  }

  /// Returns the entry's value mutably while retaining the leaf write latch.
  pub fn value_mut(&mut self) -> &mut V {
    &mut self.entry_mut().value
  }

  /// Resolves the stable index through an immutable view of the write latch.
  fn entry(&self) -> &crate::node::Entry<K, V> {
    match &self.guard.kind {
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("an entry guard must hold a leaf")
      },
      NodeKind::Leaf(leaf) => &leaf.entries[self.index],
    }
  }

  /// Resolves the stable index through the exclusive page view.
  fn entry_mut(&mut self) -> &mut crate::node::Entry<K, V> {
    match &mut self.guard.kind {
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("an entry guard must hold a leaf")
      },
      NodeKind::Leaf(leaf) => &mut leaf.entries[self.index],
    }
  }
}

impl<K, V, const N: usize> Deref for EntryMut<'_, K, V, N> {
  type Target = V;

  fn deref(&self) -> &V {
    self.value()
  }
}

impl<K, V, const N: usize> DerefMut for EntryMut<'_, K, V, N> {
  fn deref_mut(&mut self) -> &mut V {
    self.value_mut()
  }
}

impl<K: fmt::Debug, V: fmt::Debug, const N: usize> fmt::Debug
  for EntryMut<'_, K, V, N>
{
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    formatter
      .debug_struct("EntryMut")
      .field("key", self.key())
      .field("value", self.value())
      .finish()
  }
}
