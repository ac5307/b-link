use std::{
  borrow::Borrow,
  cmp::Ordering,
  mem,
  sync::{
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
    Arc, Weak,
  },
};

use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};

use crate::{array::Array, handle::KeyHandle};

/// One key/value record stored exclusively in a leaf page.
pub(crate) struct Entry<K, V> {
  /// Shared key ownership lets records, separators, and fences refer to the
  /// same allocation without requiring `K: Clone`.
  pub(crate) key: KeyHandle<K>,
  /// User value protected by the containing page latch.
  pub(crate) value: V,
}

/// Routing payload of an internal page.
///
/// `keys[i]` is the inclusive lower bound of `children[i + 1]`; therefore an
/// internal page with `n` separators always owns `n + 1` children.
pub(crate) struct InternalNode<K, V, const N: usize> {
  /// Sorted separator handles.
  keys: Array<KeyHandle<K>>,
  /// Downward pointers ordered by their half-open key ranges.
  children: Array<NodeRef<K, V, N>>,
}

/// Value-bearing payload of a leaf page.
pub(crate) struct LeafNode<K, V, const N: usize> {
  /// Sorted, unique records.
  pub(crate) entries: Array<Entry<K, V>>,
}

/// Current role of a page.
pub(crate) enum NodeKind<K, V, const N: usize> {
  /// A routing page above the leaf level.
  Internal(InternalNode<K, V, N>),
  /// A page containing user records.
  Leaf(LeafNode<K, V, N>),
  /// A logically removed page that forwards stale handles to its survivor.
  ///
  /// Parent pointers and the sibling chain are repaired before a mutation
  /// returns. The redirect remains useful for a traversal that cloned this
  /// page just before it was unlinked; `Arc` reclaims it after those stale
  /// handles and guards disappear.
  Redirect(NodeRef<K, V, N>),
}

/// One independently latched page in the concurrent tree.
///
/// Active pages own the half-open interval `[low_key, high_key)`. A missing
/// lower or upper fence represents negative or positive infinity. `right`
/// links active pages at the same level. A stale traversal that is above the
/// high fence moves right; one that is below the low fence restarts from the
/// current root. The lower-fence check is what makes right-to-left boundary
/// redistribution safe during concurrent deletion.
pub(crate) struct Node<K, V, const N: usize> {
  /// True only for the page currently published in the tree's root slot.
  /// Keeping this bit under the page latch avoids reacquiring the root-slot
  /// latch from below and therefore preserves the global latch order.
  is_root: bool,
  /// Inclusive lower fence, or negative infinity for the leftmost page.
  low_key: Option<KeyHandle<K>>,
  /// Exclusive upper fence, or positive infinity for the rightmost page.
  high_key: Option<KeyHandle<K>>,
  /// Weak predecessor used only to splice merged pages out of sibling chains.
  previous: Option<NodeWeak<K, V, N>>,
  /// Strong same-level successor used by searches and forward iterators.
  pub(crate) right: Option<NodeRef<K, V, N>>,
  /// Latched page payload inspected by entry guards and tree algorithms.
  pub(crate) kind: NodeKind<K, V, N>,
}

/// Shared page ownership used by parents, the root slot, and right links.
pub(crate) type NodeRef<K, V, const N: usize> = Arc<RwLock<Node<K, V, N>>>;

/// Non-owning predecessor pointer; using `Weak` prevents sibling-link cycles.
type NodeWeak<K, V, const N: usize> = Weak<RwLock<Node<K, V, N>>>;

/// Owning read latch whose internal `Arc` keeps its page allocated.
pub(crate) type NodeReadGuard<K, V, const N: usize> =
  ArcRwLockReadGuard<RawRwLock, Node<K, V, N>>;

/// Owning write latch used by mutations and mutable public entry guards.
pub(crate) type NodeWriteGuard<K, V, const N: usize> =
  ArcRwLockWriteGuard<RawRwLock, Node<K, V, N>>;

/// Result of publishing a preemptive split while the left page is latched.
struct Split<K, V, const N: usize> {
  /// Inclusive lower bound of the new right page.
  separator: KeyHandle<K>,
  /// Fully initialized right sibling.
  right: NodeRef<K, V, N>,
  /// Former successor whose weak back-link must be advanced to `right`.
  old_successor: Option<NodeRef<K, V, N>>,
}

/// Deferred physical sibling-chain repair after merging left into right.
struct MergeRepair<K, V, const N: usize> {
  /// Page that preceded the retired left page, when one existed.
  predecessor: Option<NodeWeak<K, V, N>>,
  /// Retired forwarding page still reachable through a stale predecessor.
  retired: NodeRef<K, V, N>,
  /// Active survivor that replaced `retired`.
  survivor: NodeRef<K, V, N>,
}

/// Owns all pages and implements traversal, insertion, deletion, and checks.
///
/// There is deliberately no tree-wide structure mutex. Structural writers use
/// top-down latch crabbing: a parent is write-latched before a full or minimally
/// occupied child and its sibling are changed. Once the child is safe, the
/// parent latch is released, so mutations in disjoint subtrees proceed at the
/// same time.
pub(crate) struct Tree<K, V, const N: usize> {
  /// Atomically published-by-latch root pointer. Traversals clone it briefly.
  root: RwLock<NodeRef<K, V, N>>,
  /// Exact number of records, updated while the destination leaf is latched.
  len: AtomicUsize,
}

/// Result of a fast insertion that either completed or retained its arguments
/// for a split-capable retry.
enum FastInsert<K, V> {
  /// The insertion or replacement completed.
  Done(Option<V>),
  /// The destination was full and neither user object was consumed.
  Structural(K, V),
}

/// Result of one optimistic top-down insertion pass.
enum InsertAttempt<K, V> {
  /// The leaf mutation completed.
  Done(Option<V>),
  /// A concurrent structural change invalidated the pass; retry from root.
  Retry(K, V),
}

/// Result of the single-leaf deletion fast path.
enum FastRemove<V> {
  /// The lookup/removal completed without threatening minimum occupancy.
  Done(Option<V>),
  /// The key exists in a minimally occupied non-root leaf.
  Structural,
}

/// Result of one top-down deletion pass.
enum RemoveAttempt<V> {
  /// The lookup/removal completed.
  Done(Option<V>),
  /// A raced page became unsafe, so a new pass must repair it from its parent.
  Retry,
}

/// Direction selected by the two-fence stale-route check.
enum Route<K, V, const N: usize> {
  /// The key is below the page's new lower fence; restart at the current root.
  Restart,
  /// Follow a same-level successor or a retired-page forwarding pointer.
  Right(NodeRef<K, V, N>),
  /// This active page owns the key's range.
  Here,
}

impl<K, V, const N: usize> Node<K, V, N> {
  /// Allocates the initial empty leaf covering the complete key space.
  fn leaf_root() -> NodeRef<K, V, N> {
    Arc::new(RwLock::new(Self {
      is_root: true,
      low_key: None,
      high_key: None,
      previous: None,
      right: None,
      kind: NodeKind::Leaf(LeafNode {
        entries: Array::with_capacity(N),
      }),
    }))
  }

  /// Builds a new root above a preemptively split former root.
  fn root_from_split(
    left: NodeRef<K, V, N>,
    separator: KeyHandle<K>,
    right: NodeRef<K, V, N>,
  ) -> NodeRef<K, V, N> {
    let mut keys = Array::with_capacity(N);
    keys.push(separator);

    let mut children = Array::with_capacity(N + 1);
    children.push(left);
    children.push(right);

    Arc::new(RwLock::new(Self {
      is_root: true,
      low_key: None,
      high_key: None,
      previous: None,
      right: None,
      kind: NodeKind::Internal(InternalNode { keys, children }),
    }))
  }

  /// Returns the number of records or separators in an active page.
  fn occupancy(&self) -> usize {
    match &self.kind {
      NodeKind::Internal(internal) => internal.keys.len(),
      NodeKind::Leaf(leaf) => leaf.entries.len(),
      NodeKind::Redirect(_) => 0,
    }
  }

  /// Returns the stable minimum occupancy for this page kind.
  fn minimum(&self) -> usize {
    match self.kind {
      NodeKind::Internal(_) => (N - 1) / 2,
      NodeKind::Leaf(_) => N / 2,
      NodeKind::Redirect(_) => 0,
    }
  }

  /// Classifies a key against the page's current role and half-open fences.
  fn route<Q>(&self, key: &Q) -> Route<K, V, N>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    if let NodeKind::Redirect(target) = &self.kind {
      return Route::Right(Arc::clone(target));
    }

    if self.low_key.as_ref().is_some_and(|low_key| {
      key.cmp(low_key.get().borrow()) == Ordering::Less
    }) {
      return Route::Restart;
    }

    if self.high_key.as_ref().is_some_and(|high_key| {
      key.cmp(high_key.get().borrow()) != Ordering::Less
    }) {
      let right = self
        .right
        .as_ref()
        .expect("a finite high fence must have a right sibling");
      return Route::Right(Arc::clone(right));
    }

    Route::Here
  }

  /// Splits a stable full page before another record/separator is inserted.
  ///
  /// The caller holds the page's write latch and its parent latch (unless this
  /// is the root). The right page is fully initialized before `self.high_key`
  /// and `self.right` publish it, so a stale route is correct even before the
  /// parent receives the separator.
  fn split_full(&mut self, left_ref: &NodeRef<K, V, N>) -> Split<K, V, N> {
    debug_assert_eq!(self.occupancy(), N);

    let (separator, right_kind) = match &mut self.kind {
      NodeKind::Leaf(leaf) => {
        // Leaves retain the lower `floor(N / 2)` records and move the rest.
        let right_entries = leaf.entries.split_off(N / 2);
        let separator = right_entries
          .first()
          .expect("a full leaf split creates a non-empty right page")
          .key
          .clone();
        (
          separator,
          NodeKind::Leaf(LeafNode {
            entries: right_entries,
          }),
        )
      },
      NodeKind::Internal(internal) => {
        // The middle separator is promoted. Children above it move right.
        let middle = N / 2;
        let separator = internal.keys.remove(middle);
        let right_keys = internal.keys.split_off(middle);
        let right_children = internal.children.split_off(middle + 1);

        debug_assert_eq!(internal.children.len(), internal.keys.len() + 1);
        debug_assert_eq!(right_children.len(), right_keys.len() + 1);

        (
          separator,
          NodeKind::Internal(InternalNode {
            keys: right_keys,
            children: right_children,
          }),
        )
      },
      NodeKind::Redirect(_) => unreachable!("cannot split a retired page"),
    };

    let old_successor = self.right.take();
    let right = Arc::new(RwLock::new(Self {
      is_root: false,
      low_key: Some(separator.clone()),
      high_key: self.high_key.take(),
      previous: Some(Arc::downgrade(left_ref)),
      right: old_successor.as_ref().map(Arc::clone),
      kind: right_kind,
    }));

    // Publish the escape path while the left page is still exclusively
    // latched. Parent insertion is an index optimization that may lag.
    self.high_key = Some(separator.clone());
    self.right = Some(Arc::clone(&right));

    Split {
      separator,
      right,
      old_successor,
    }
  }
}

impl<K: Ord, V, const N: usize> InternalNode<K, V, N> {
  /// Finds the child whose half-open interval contains `key`.
  ///
  /// This upper-bound search advances on equality because a separator is the
  /// inclusive lower bound of the child immediately to its right.
  fn child_index<Q>(&self, key: &Q) -> usize
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let mut low = 0;
    let mut high = self.keys.len();

    while low < high {
      let middle = low + (high - low) / 2;

      if self.keys[middle].get().borrow().cmp(key) == Ordering::Greater {
        high = middle;
      } else {
        low = middle + 1;
      }
    }

    low
  }

  /// Installs a newly split sibling immediately after its left child.
  fn insert_split(
    &mut self,
    child_index: usize,
    separator: KeyHandle<K>,
    right: NodeRef<K, V, N>,
  ) {
    debug_assert!(self.keys.len() < N);
    debug_assert_eq!(self.children.len(), self.keys.len() + 1);
    debug_assert!(child_index < self.children.len());

    self.keys.insert(child_index, separator);
    self.children.insert(child_index + 1, right);
  }
}

impl<K: Ord, V, const N: usize> LeafNode<K, V, N> {
  /// Binary-searches the leaf for a key or its stable insertion position.
  pub(crate) fn search<Q>(&self, key: &Q) -> Result<usize, usize>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    self
      .entries
      .as_slice()
      .binary_search_by(|entry| entry.key.get().borrow().cmp(key))
  }

  /// Inserts or replaces one record while the containing page is write-latched.
  fn insert(&mut self, key: K, value: V) -> (Option<V>, bool) {
    match self.search(&key) {
      Ok(index) => {
        let old_value = mem::replace(&mut self.entries[index].value, value);
        (Some(old_value), false)
      },
      Err(index) => {
        self.entries.insert(
          index,
          Entry {
            key: KeyHandle::new(key),
            value,
          },
        );
        (None, true)
      },
    }
  }
}

impl<K, V, const N: usize> Tree<K, V, N> {
  /// Creates a tree containing one empty root leaf.
  pub(crate) fn new() -> Self {
    // Stable pages must split into two non-empty pages. Checked arithmetic
    // protects the internal page's `N + 1` child-pointer allocation.
    assert!(N >= 3, "BLinkMap page capacity must be at least 3");
    assert!(N.checked_add(1).is_some(), "BLinkMap capacity overflow");

    Self {
      root: RwLock::new(Node::leaf_root()),
      len: AtomicUsize::new(0),
    }
  }

  /// Loads the exact record counter without taking a page latch.
  pub(crate) fn len(&self) -> usize {
    self.len.load(AtomicOrdering::Relaxed)
  }

  /// Clones the current root pointer and immediately releases the root latch.
  fn root_node(&self) -> NodeRef<K, V, N> {
    Arc::clone(&self.root.read())
  }
}

impl<K: Ord, V, const N: usize> Tree<K, V, N> {
  /// Finds `key` and returns a read-latched leaf plus its stable record index.
  pub(crate) fn get<Q>(
    &self,
    key: &Q,
  ) -> Option<(NodeReadGuard<K, V, N>, usize)>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (_, guard) = self.leaf_read_for(key);
    let index = match &guard.kind {
      NodeKind::Leaf(leaf) => leaf.search(key).ok()?,
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("leaf traversal returned a non-leaf page")
      },
    };

    Some((guard, index))
  }

  /// Finds `key` and returns a write-latched leaf plus its stable record index.
  pub(crate) fn get_mut<Q>(
    &self,
    key: &Q,
  ) -> Option<(NodeWriteGuard<K, V, N>, usize)>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (_, guard) = self.leaf_write_for(key);
    let index = match &guard.kind {
      NodeKind::Leaf(leaf) => leaf.search(key).ok()?,
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("leaf traversal returned a non-leaf page")
      },
    };

    Some((guard, index))
  }

  /// Returns the first record in key order for iterator and boundary APIs.
  pub(crate) fn first_entry(
    &self,
  ) -> Option<(NodeReadGuard<K, V, N>, usize)> {
    let mut node = self.leftmost_leaf();

    loop {
      let guard = read_node(&node);
      match &guard.kind {
        NodeKind::Leaf(leaf) if !leaf.entries.is_empty() => {
          return Some((guard, 0));
        },
        NodeKind::Leaf(_) => {
          let next = guard.right.as_ref().map(Arc::clone)?;
          drop(guard);
          node = next;
        },
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          node = target;
        },
        NodeKind::Internal(_) => {
          unreachable!("leftmost-leaf traversal returned an internal page")
        },
      }
    }
  }

  /// Returns the last record in key order for boundary APIs.
  pub(crate) fn last_entry(
    &self,
  ) -> Option<(NodeReadGuard<K, V, N>, usize)> {
    let mut node = self.rightmost_leaf();

    loop {
      let guard = read_node(&node);
      match &guard.kind {
        NodeKind::Leaf(leaf) if !leaf.entries.is_empty() => {
          let index = leaf.entries.len() - 1;
          return Some((guard, index));
        },
        NodeKind::Leaf(_) => return None,
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          node = target;
        },
        NodeKind::Internal(_) => {
          unreachable!("rightmost-leaf traversal returned an internal page")
        },
      }
    }
  }

  /// Returns the least record satisfying an inclusive or exclusive bound.
  pub(crate) fn entry_at_or_after<Q>(
    &self,
    key: &Q,
    inclusive: bool,
  ) -> Option<(NodeReadGuard<K, V, N>, usize)>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (_, mut guard) = self.leaf_read_for(key);

    loop {
      match &guard.kind {
        NodeKind::Leaf(leaf) => {
          let index = match leaf.search(key) {
            Ok(index) if inclusive => index,
            Ok(index) => index + 1,
            Err(index) => index,
          };

          if index < leaf.entries.len() {
            return Some((guard, index));
          }

          let next = guard.right.as_ref().map(Arc::clone)?;
          drop(guard);
          guard = read_node(&next);
          continue;
        },
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          guard = read_node(&target);
          continue;
        },
        NodeKind::Internal(_) => {
          unreachable!("leaf traversal returned an internal page")
        },
      }
    }
  }

  /// Inserts or replaces one record without a global topology lock.
  pub(crate) fn insert(&self, key: K, value: V) -> Option<V> {
    let (mut key, mut value) =
      match self.try_insert_without_split(key, value) {
        FastInsert::Done(replaced) => return replaced,
        FastInsert::Structural(key, value) => (key, value),
      };

    loop {
      // A full root has no parent that could split it during the descent, so
      // publish a new root first. Root-slot locking is intentionally brief.
      self.split_root_if_full();

      match self.insert_attempt(key, value) {
        InsertAttempt::Done(replaced) => return replaced,
        InsertAttempt::Retry(retry_key, retry_value) => {
          key = retry_key;
          value = retry_value;
          // Avoid monopolizing a CPU when a public leaf guard is deliberately
          // retained by another thread.
          std::thread::yield_now();
        },
      }
    }
  }

  /// Removes one record and repairs every minimum-occupancy violation before
  /// descending into it.
  pub(crate) fn remove<Q>(&self, key: &Q) -> Option<V>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    match self.try_remove_without_rebalance(key) {
      FastRemove::Done(removed) => return removed,
      FastRemove::Structural => {},
    }

    loop {
      self.shrink_root();

      match self.remove_attempt(key) {
        RemoveAttempt::Done(removed) => {
          self.shrink_root();
          return removed;
        },
        RemoveAttempt::Retry => std::thread::yield_now(),
      }
    }
  }

  /// Attempts a replacement or a non-splitting insert under one leaf latch.
  fn try_insert_without_split(&self, key: K, value: V) -> FastInsert<K, V> {
    let (_, mut guard) = self.leaf_write_for(&key);
    let leaf = match &mut guard.kind {
      NodeKind::Leaf(leaf) => leaf,
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("leaf traversal returned a non-leaf page")
      },
    };

    if leaf.search(&key).is_err() && leaf.entries.len() == N {
      return FastInsert::Structural(key, value);
    }

    let (replaced, inserted) = leaf.insert(key, value);
    if inserted {
      // Publish the count while the record is still protected by the leaf
      // latch, giving insert/remove a consistent linearization order.
      self.len.fetch_add(1, AtomicOrdering::Relaxed);
    }

    FastInsert::Done(replaced)
  }

  /// Attempts a removal that leaves the destination at or above its minimum.
  fn try_remove_without_rebalance<Q>(&self, key: &Q) -> FastRemove<V>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let (_, mut guard) = self.leaf_write_for(key);
    let is_root = guard.is_root;
    let leaf = match &mut guard.kind {
      NodeKind::Leaf(leaf) => leaf,
      NodeKind::Internal(_) | NodeKind::Redirect(_) => {
        unreachable!("leaf traversal returned a non-leaf page")
      },
    };

    let index = match leaf.search(key) {
      Ok(index) => index,
      Err(_) => return FastRemove::Done(None),
    };

    if !is_root && leaf.entries.len() <= N / 2 {
      return FastRemove::Structural;
    }

    let entry = leaf.entries.remove(index);
    self.decrement_len();
    FastRemove::Done(Some(entry.value))
  }

  /// Runs one optimistic top-down insertion pass.
  fn insert_attempt(&self, key: K, value: V) -> InsertAttempt<K, V> {
    let mut node = self.root_node();

    loop {
      let mut guard = write_node(&node);

      match guard.route(&key) {
        Route::Restart => return InsertAttempt::Retry(key, value),
        Route::Right(next) => {
          drop(guard);
          node = next;
          continue;
        },
        Route::Here => {},
      }

      // A page can become full after its parent was released because another
      // writer reached it first. Retrying lets this pass split it while holding
      // that parent rather than violating the top-down latch order.
      if guard.occupancy() == N {
        return InsertAttempt::Retry(key, value);
      }

      match &mut guard.kind {
        NodeKind::Leaf(leaf) => {
          let (replaced, inserted) = leaf.insert(key, value);
          if inserted {
            self.len.fetch_add(1, AtomicOrdering::Relaxed);
          }
          return InsertAttempt::Done(replaced);
        },
        NodeKind::Internal(internal) => {
          let child_index = internal.child_index(&key);
          let child_ref = Arc::clone(&internal.children[child_index]);
          let Some(mut child) = try_write_node(&child_ref) else {
            return InsertAttempt::Retry(key, value);
          };

          // Parent ownership and its write latch make this child-array slot
          // stable. A redirect here would indicate a raced stale parent.
          if matches!(child.kind, NodeKind::Redirect(_)) {
            return InsertAttempt::Retry(key, value);
          }

          let mut repair = None;
          let target = if child.occupancy() == N {
            let split = child.split_full(&child_ref);
            let goes_right =
              key.cmp(split.separator.get()) != Ordering::Less;
            let target = if goes_right {
              Arc::clone(&split.right)
            } else {
              Arc::clone(&child_ref)
            };

            internal.insert_split(
              child_index,
              split.separator,
              Arc::clone(&split.right),
            );
            repair = Some((split.right, split.old_successor));
            target
          } else {
            Arc::clone(&child_ref)
          };

          drop(child);
          drop(guard);

          if let Some((new_right, old_successor)) = repair {
            repair_split_backlink(&new_right, old_successor);
          }

          node = target;
        },
        NodeKind::Redirect(_) => unreachable!("route handles redirects"),
      }
    }
  }

  /// Runs one top-down deletion pass, borrowing or merging before descent.
  fn remove_attempt<Q>(&self, key: &Q) -> RemoveAttempt<V>
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    let mut node = self.root_node();

    loop {
      let mut guard = write_node(&node);

      match guard.route(key) {
        Route::Restart => return RemoveAttempt::Retry,
        Route::Right(next) => {
          drop(guard);
          node = next;
          continue;
        },
        Route::Here => {},
      }

      if !guard.is_root && guard.occupancy() <= guard.minimum() {
        // A concurrent pass changed this page after its parent latch was
        // released. Restart so its parent can first make it deletion-safe.
        return RemoveAttempt::Retry;
      }

      match &mut guard.kind {
        NodeKind::Leaf(leaf) => {
          let index = match leaf.search(key) {
            Ok(index) => index,
            Err(_) => return RemoveAttempt::Done(None),
          };
          let entry = leaf.entries.remove(index);
          self.decrement_len();
          return RemoveAttempt::Done(Some(entry.value));
        },
        NodeKind::Internal(internal) => {
          let child_index = internal.child_index(key);

          if child_index > 0 {
            // Structural writers acquire adjacent siblings from left to right.
            let left_ref = Arc::clone(&internal.children[child_index - 1]);
            let child_ref = Arc::clone(&internal.children[child_index]);
            let Some(mut left) = try_write_node(&left_ref) else {
              return RemoveAttempt::Retry;
            };
            let Some(mut child) = try_write_node(&child_ref) else {
              return RemoveAttempt::Retry;
            };

            if matches!(left.kind, NodeKind::Redirect(_))
              || matches!(child.kind, NodeKind::Redirect(_))
            {
              return RemoveAttempt::Retry;
            }

            let child_minimum = child.minimum();
            let mut merge_repair = None;

            if child.occupancy() <= child_minimum {
              if left.occupancy() > left.minimum() {
                borrow_from_left(
                  internal,
                  child_index - 1,
                  &mut left,
                  &mut child,
                );
              } else {
                merge_repair = Some(merge_left_into_right(
                  internal,
                  child_index - 1,
                  &left_ref,
                  &mut left,
                  &child_ref,
                  &mut child,
                ));
              }
            }

            drop(child);
            drop(left);
            drop(guard);

            if let Some(repair) = merge_repair {
              repair_merged_link(repair);
            }

            node = child_ref;
          } else {
            // The first child has no left donor. Pair it with its right sibling.
            if internal.children.len() < 2 {
              return RemoveAttempt::Retry;
            }

            let child_ref = Arc::clone(&internal.children[0]);
            let right_ref = Arc::clone(&internal.children[1]);
            let Some(mut child) = try_write_node(&child_ref) else {
              return RemoveAttempt::Retry;
            };
            let Some(mut right) = try_write_node(&right_ref) else {
              return RemoveAttempt::Retry;
            };

            if matches!(child.kind, NodeKind::Redirect(_))
              || matches!(right.kind, NodeKind::Redirect(_))
            {
              return RemoveAttempt::Retry;
            }

            let child_minimum = child.minimum();
            let mut merge_repair = None;
            let target;

            if child.occupancy() <= child_minimum {
              if right.occupancy() > right.minimum() {
                borrow_from_right(internal, 0, &mut child, &mut right);
                target = Arc::clone(&child_ref);
              } else {
                merge_repair = Some(merge_left_into_right(
                  internal, 0, &child_ref, &mut child, &right_ref,
                  &mut right,
                ));
                target = Arc::clone(&right_ref);
              }
            } else {
              target = Arc::clone(&child_ref);
            }

            drop(right);
            drop(child);
            drop(guard);

            if let Some(repair) = merge_repair {
              repair_merged_link(repair);
            }

            node = target;
          }
        },
        NodeKind::Redirect(_) => unreachable!("route handles redirects"),
      }
    }
  }

  /// Splits a full root while holding only the root-slot and root-page latches.
  fn split_root_if_full(&self) {
    loop {
      let mut root_slot = self.root.write();
      let root_ref = Arc::clone(&root_slot);
      let mut root = write_node(&root_ref);

      if let NodeKind::Redirect(target) = &root.kind {
        *root_slot = Arc::clone(target);
        continue;
      }

      if root.occupancy() < N {
        return;
      }

      let split = root.split_full(&root_ref);
      root.is_root = false;
      let old_successor = split.old_successor;
      let right = Arc::clone(&split.right);
      *root_slot = Node::root_from_split(
        Arc::clone(&root_ref),
        split.separator,
        split.right,
      );

      drop(root);
      drop(root_slot);
      repair_split_backlink(&right, old_successor);
      return;
    }
  }

  /// Removes empty internal root levels after a child merge.
  fn shrink_root(&self) {
    loop {
      let mut root_slot = self.root.write();
      let root_ref = Arc::clone(&root_slot);
      let mut root = write_node(&root_ref);

      let child = match &root.kind {
        NodeKind::Internal(internal) if internal.keys.is_empty() => {
          debug_assert_eq!(internal.children.len(), 1);
          Arc::clone(&internal.children[0])
        },
        NodeKind::Redirect(target) => Arc::clone(target),
        NodeKind::Internal(_) | NodeKind::Leaf(_) => return,
      };

      let mut child_guard = write_node(&child);
      // A sole root child covers the complete key space and has no active peer
      // at its level. Clear inherited fences before publishing it as root.
      child_guard.low_key = None;
      child_guard.high_key = None;
      child_guard.previous = None;
      child_guard.right = None;
      child_guard.is_root = true;

      root.is_root = false;
      root.kind = NodeKind::Redirect(Arc::clone(&child));
      root.low_key = None;
      root.high_key = None;
      root.previous = None;
      root.right = None;
      *root_slot = child;

      // Repeat because a single deletion can expose several empty root levels.
    }
  }

  /// Traverses read-latched pages until an active leaf owns `key`.
  fn leaf_read_for<Q>(
    &self,
    key: &Q,
  ) -> (NodeRef<K, V, N>, NodeReadGuard<K, V, N>)
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    'restart: loop {
      let mut node = self.root_node();

      loop {
        let guard = read_node(&node);

        match guard.route(key) {
          Route::Restart => continue 'restart,
          Route::Right(next) => {
            drop(guard);
            node = next;
          },
          Route::Here => match &guard.kind {
            NodeKind::Internal(internal) => {
              let child =
                Arc::clone(&internal.children[internal.child_index(key)]);
              drop(guard);
              node = child;
            },
            NodeKind::Leaf(_) => return (node, guard),
            NodeKind::Redirect(_) => unreachable!("route handles redirects"),
          },
        }
      }
    }
  }

  /// Descends through first-child pointers to an active leaf.
  fn leftmost_leaf(&self) -> NodeRef<K, V, N> {
    let mut node = self.root_node();

    loop {
      let guard = read_node(&node);
      match &guard.kind {
        NodeKind::Internal(internal) => {
          let child = Arc::clone(&internal.children[0]);
          drop(guard);
          node = child;
        },
        NodeKind::Leaf(_) => return node,
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          node = target;
        },
      }
    }
  }

  /// Descends through last-child pointers, then follows any split links that
  /// appeared after the root snapshot, to reach a rightmost leaf.
  fn rightmost_leaf(&self) -> NodeRef<K, V, N> {
    let mut node = self.root_node();

    loop {
      let guard = read_node(&node);
      match &guard.kind {
        NodeKind::Internal(internal) => {
          let child = Arc::clone(
            internal
              .children
              .last()
              .expect("an internal page always has a child"),
          );
          drop(guard);
          node = child;
        },
        NodeKind::Leaf(_) => {
          let Some(next) = guard.right.as_ref().map(Arc::clone) else {
            return node;
          };
          drop(guard);
          node = next;
        },
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          node = target;
        },
      }
    }
  }

  /// Traverses with read latches, then reacquires and revalidates the leaf with
  /// an exclusive latch. No ancestor latch is retained while waiting.
  fn leaf_write_for<Q>(
    &self,
    key: &Q,
  ) -> (NodeRef<K, V, N>, NodeWriteGuard<K, V, N>)
  where
    K: Borrow<Q>,
    Q: Ord + ?Sized,
  {
    loop {
      let (node, read_guard) = self.leaf_read_for(key);
      drop(read_guard);

      let guard = write_node(&node);
      match guard.route(key) {
        Route::Here if matches!(guard.kind, NodeKind::Leaf(_)) => {
          return (node, guard);
        },
        Route::Restart | Route::Right(_) | Route::Here => {},
      }
    }
  }

  /// Decrements the exact length while the removed record's leaf is latched.
  fn decrement_len(&self) {
    let previous = self.len.fetch_sub(1, AtomicOrdering::Relaxed);
    debug_assert!(previous > 0);
  }

  /// Returns the number of active page levels.
  #[cfg(test)]
  pub(crate) fn height(&self) -> usize {
    let mut height = 1;
    let mut node = self.root_node();

    loop {
      let guard = read_node(&node);
      match &guard.kind {
        NodeKind::Internal(internal) => {
          let child = Arc::clone(&internal.children[0]);
          drop(guard);
          node = child;
          height += 1;
        },
        NodeKind::Leaf(_) => return height,
        NodeKind::Redirect(target) => {
          let target = Arc::clone(target);
          drop(guard);
          node = target;
        },
      }
    }
  }

  /// Performs an exhaustive quiescent invariant check used by tests.
  #[cfg(test)]
  pub(crate) fn validate(&self) -> Result<(), String> {
    let root = self.root_node();
    let root_guard = read_node(&root);

    if root_guard.low_key.is_some()
      || root_guard.high_key.is_some()
      || root_guard.previous.is_some()
      || root_guard.right.is_some()
    {
      return Err(String::from("root must cover the complete key space"));
    }
    if !root_guard.is_root {
      return Err(String::from("published root is not marked as root"));
    }
    if matches!(root_guard.kind, NodeKind::Redirect(_)) {
      return Err(String::from("root must not be a redirect"));
    }
    drop(root_guard);

    let mut leaf_depth = None;
    let mut counted = 0;
    validate_node(&root, true, 1, &mut leaf_depth, &mut counted)?;

    if counted != self.len() {
      return Err(format!(
        "entry counter mismatch: leaves contain {counted}, len is {}",
        self.len()
      ));
    }

    validate_sibling_levels(&root, self.len(), self.height())
  }
}

/// Moves one record/edge from `left` into the beginning of `right`.
fn borrow_from_left<K: Ord, V, const N: usize>(
  parent: &mut InternalNode<K, V, N>,
  separator_index: usize,
  left: &mut Node<K, V, N>,
  right: &mut Node<K, V, N>,
) {
  let boundary = match (&mut left.kind, &mut right.kind) {
    (NodeKind::Leaf(left_leaf), NodeKind::Leaf(right_leaf)) => {
      let entry = left_leaf
        .entries
        .pop()
        .expect("a lending leaf must contain a record");
      let boundary = entry.key.clone();
      right_leaf.entries.insert(0, entry);
      parent.keys[separator_index] = boundary.clone();
      boundary
    },
    (
      NodeKind::Internal(left_internal),
      NodeKind::Internal(right_internal),
    ) => {
      let boundary = left_internal
        .keys
        .pop()
        .expect("a lending internal page must contain a separator");
      let old_parent =
        mem::replace(&mut parent.keys[separator_index], boundary.clone());
      let child = left_internal
        .children
        .pop()
        .expect("an internal page must retain one more child than key");
      right_internal.keys.insert(0, old_parent);
      right_internal.children.insert(0, child);
      boundary
    },
    _ => unreachable!("siblings must have the same page kind"),
  };

  // Boundary metadata and the parent separator change while both sibling
  // latches are held, so a later stale traversal observes a complete version.
  left.high_key = Some(boundary.clone());
  right.low_key = Some(boundary);
}

/// Moves one record/edge from `right` onto the end of `left`.
fn borrow_from_right<K: Ord, V, const N: usize>(
  parent: &mut InternalNode<K, V, N>,
  separator_index: usize,
  left: &mut Node<K, V, N>,
  right: &mut Node<K, V, N>,
) {
  let boundary = match (&mut left.kind, &mut right.kind) {
    (NodeKind::Leaf(left_leaf), NodeKind::Leaf(right_leaf)) => {
      let entry = right_leaf.entries.remove(0);
      left_leaf.entries.push(entry);
      let boundary = right_leaf
        .entries
        .first()
        .expect("a lending right leaf remains non-empty")
        .key
        .clone();
      parent.keys[separator_index] = boundary.clone();
      boundary
    },
    (
      NodeKind::Internal(left_internal),
      NodeKind::Internal(right_internal),
    ) => {
      let boundary = right_internal.keys.remove(0);
      let old_parent =
        mem::replace(&mut parent.keys[separator_index], boundary.clone());
      let child = right_internal.children.remove(0);
      left_internal.keys.push(old_parent);
      left_internal.children.push(child);
      boundary
    },
    _ => unreachable!("siblings must have the same page kind"),
  };

  left.high_key = Some(boundary.clone());
  right.low_key = Some(boundary);
}

/// Merges an adjacent left page into its right sibling and retires the left.
///
/// Keys move toward a surviving right page. A stale handle to the left sees a
/// forwarding state while the parent's left child pointer is removed.
fn merge_left_into_right<K: Ord, V, const N: usize>(
  parent: &mut InternalNode<K, V, N>,
  left_index: usize,
  left_ref: &NodeRef<K, V, N>,
  left: &mut Node<K, V, N>,
  right_ref: &NodeRef<K, V, N>,
  right: &mut Node<K, V, N>,
) -> MergeRepair<K, V, N> {
  debug_assert!(Arc::ptr_eq(&parent.children[left_index], left_ref));
  debug_assert!(Arc::ptr_eq(&parent.children[left_index + 1], right_ref));

  let separator = parent.keys.remove(left_index);
  let removed_child = parent.children.remove(left_index);
  debug_assert!(Arc::ptr_eq(&removed_child, left_ref));

  match (&mut left.kind, &mut right.kind) {
    (NodeKind::Leaf(left_leaf), NodeKind::Leaf(right_leaf)) => {
      debug_assert!(left_leaf.entries.len() + right_leaf.entries.len() <= N);
      right_leaf.entries.prepend_from(&mut left_leaf.entries);
      // Leaf separators are duplicated metadata, so removing the parent copy
      // is sufficient; `separator` drops at the end of this arm's scope.
      drop(separator);
    },
    (
      NodeKind::Internal(left_internal),
      NodeKind::Internal(right_internal),
    ) => {
      debug_assert!(
        left_internal.keys.len() + 1 + right_internal.keys.len() <= N
      );
      right_internal.keys.insert(0, separator);
      right_internal.keys.prepend_from(&mut left_internal.keys);
      right_internal
        .children
        .prepend_from(&mut left_internal.children);
    },
    _ => unreachable!("siblings must have the same page kind"),
  }

  let predecessor = left.previous.clone();
  right.low_key = left.low_key.take();
  right.previous = predecessor.clone();

  // Publish forwarding before releasing metadata handles. If dropping a key
  // destructor unwinds, every stale route still reaches the survivor.
  left.kind = NodeKind::Redirect(Arc::clone(right_ref));

  MergeRepair {
    predecessor,
    retired: Arc::clone(left_ref),
    survivor: Arc::clone(right_ref),
  }
}

/// Advances a split successor's weak predecessor after structural latches drop.
fn repair_split_backlink<K, V, const N: usize>(
  new_right: &NodeRef<K, V, N>,
  old_successor: Option<NodeRef<K, V, N>>,
) {
  let Some(mut successor) = old_successor else {
    return;
  };

  let mut right_guard = write_node(new_right);
  if matches!(right_guard.kind, NodeKind::Redirect(_)) {
    return;
  }

  // This repair intentionally runs after releasing ancestors. If `new_right`
  // has already split or been promoted, a newer operation owns its successor
  // metadata and this stale repair must not overwrite it.
  let still_points_to_original = right_guard
    .right
    .as_ref()
    .is_some_and(|right| Arc::ptr_eq(right, &successor));
  if !still_points_to_original {
    return;
  }

  loop {
    let mut successor_guard = write_node(&successor);
    if let NodeKind::Redirect(target) = &successor_guard.kind {
      let target = Arc::clone(target);
      right_guard.right = Some(Arc::clone(&target));
      drop(successor_guard);
      successor = target;
      continue;
    }

    successor_guard.previous = Some(Arc::downgrade(new_right));
    right_guard.right = Some(Arc::clone(&successor));
    return;
  }
}

/// Splices a retired merged page out of the forward sibling chain.
fn repair_merged_link<K, V, const N: usize>(repair: MergeRepair<K, V, N>) {
  let Some(mut cursor) = repair.predecessor.as_ref().and_then(Weak::upgrade)
  else {
    let mut survivor = write_node(&repair.survivor);
    survivor.previous = None;
    return;
  };

  // A predecessor can split after the merger publishes its redirect and before
  // this repair runs. Walk forward from the captured predecessor, compacting
  // any other redirects encountered, until the exact incoming link is found.
  // Every latch acquisition moves right, matching the structural latch order.
  loop {
    let mut cursor_guard = write_node(&cursor);

    if let NodeKind::Redirect(_) = cursor_guard.kind {
      // This predecessor was itself retired first. Its retained weak back-link
      // gives us an earlier point from which to repeat the forward search.
      let earlier = cursor_guard.previous.as_ref().and_then(Weak::upgrade);
      let target = match &cursor_guard.kind {
        NodeKind::Redirect(target) => Arc::clone(target),
        NodeKind::Internal(_) | NodeKind::Leaf(_) => unreachable!(),
      };
      drop(cursor_guard);
      cursor = earlier.unwrap_or(target);
      continue;
    }

    let Some(next) = cursor_guard.right.as_ref().map(Arc::clone) else {
      // The only legitimate missing predecessor is the new leftmost survivor.
      drop(cursor_guard);
      let mut survivor = write_node(&repair.survivor);
      survivor.previous = None;
      return;
    };

    if Arc::ptr_eq(&next, &repair.retired) {
      cursor_guard.right = Some(Arc::clone(&repair.survivor));
      let mut survivor = write_node(&repair.survivor);

      if let NodeKind::Redirect(target) = &survivor.kind {
        let target = Arc::clone(target);
        cursor_guard.right = Some(Arc::clone(&target));
        drop(survivor);
        drop(cursor_guard);
        cursor = target;
        continue;
      }

      survivor.previous = Some(Arc::downgrade(&cursor));
      return;
    }

    if Arc::ptr_eq(&next, &repair.survivor) {
      let mut survivor = write_node(&repair.survivor);
      if let NodeKind::Redirect(target) = &survivor.kind {
        cursor_guard.right = Some(Arc::clone(target));
        continue;
      }
      survivor.previous = Some(Arc::downgrade(&cursor));
      return;
    }

    let next_guard = read_node(&next);
    if let NodeKind::Redirect(target) = &next_guard.kind {
      let target = Arc::clone(target);
      cursor_guard.right = Some(Arc::clone(&target));
      drop(next_guard);

      if Arc::ptr_eq(&target, &repair.survivor) {
        let mut survivor = write_node(&repair.survivor);
        if !matches!(survivor.kind, NodeKind::Redirect(_)) {
          survivor.previous = Some(Arc::downgrade(&cursor));
          return;
        }
      }

      drop(cursor_guard);
      continue;
    }

    drop(next_guard);
    drop(cursor_guard);
    cursor = next;
  }
}

/// Acquires an owning read latch for a shared page reference.
pub(crate) fn read_node<K, V, const N: usize>(
  node: &NodeRef<K, V, N>,
) -> NodeReadGuard<K, V, N> {
  node.read_arc()
}

/// Acquires an owning write latch for a shared page reference.
fn write_node<K, V, const N: usize>(
  node: &NodeRef<K, V, N>,
) -> NodeWriteGuard<K, V, N> {
  node.write_arc()
}

/// Attempts to acquire an owning write latch without waiting under a parent.
fn try_write_node<K, V, const N: usize>(
  node: &NodeRef<K, V, N>,
) -> Option<NodeWriteGuard<K, V, N>> {
  node.try_write_arc()
}

/// Recursively validates parent ownership, ordering, fences, occupancy, depth,
/// and the aggregate record count. Tests call this only after writers join.
#[cfg(test)]
fn validate_node<K: Ord, V, const N: usize>(
  node: &NodeRef<K, V, N>,
  is_root: bool,
  depth: usize,
  leaf_depth: &mut Option<usize>,
  record_count: &mut usize,
) -> Result<(), String> {
  let guard = read_node(node);

  if guard.occupancy() > N {
    return Err(String::from("page exceeds stable capacity"));
  }
  if guard.is_root != is_root {
    return Err(String::from("page root-role marker is inconsistent"));
  }
  if !is_root && guard.occupancy() < guard.minimum() {
    return Err(String::from("non-root page is below minimum occupancy"));
  }

  match &guard.kind {
    NodeKind::Redirect(_) => {
      return Err(String::from("parent tree reaches a retired page"));
    },
    NodeKind::Leaf(leaf) => {
      if let Some(expected) = *leaf_depth {
        if expected != depth {
          return Err(String::from("leaves do not share one depth"));
        }
      } else {
        *leaf_depth = Some(depth);
      }

      for pair in leaf.entries.as_slice().windows(2) {
        if pair[0].key >= pair[1].key {
          return Err(String::from("leaf records are not strictly ordered"));
        }
      }
      for entry in leaf.entries.as_slice() {
        if !within_fences(&entry.key, &guard.low_key, &guard.high_key) {
          return Err(String::from("leaf record lies outside page fences"));
        }
      }
      *record_count += leaf.entries.len();
    },
    NodeKind::Internal(internal) => {
      if internal.children.len() != internal.keys.len() + 1 {
        return Err(String::from("internal child/key cardinality mismatch"));
      }
      if is_root && internal.keys.is_empty() {
        return Err(String::from("empty internal root was not collapsed"));
      }
      for pair in internal.keys.as_slice().windows(2) {
        if pair[0] >= pair[1] {
          return Err(String::from("internal separators are not ordered"));
        }
      }
      for key in internal.keys.as_slice() {
        if !within_fences(key, &guard.low_key, &guard.high_key) {
          return Err(String::from("separator lies outside page fences"));
        }
      }

      for child_index in 0..internal.children.len() {
        let child = &internal.children[child_index];
        let child_guard = read_node(child);
        let expected_low = if child_index == 0 {
          guard.low_key.as_ref()
        } else {
          Some(&internal.keys[child_index - 1])
        };
        let expected_high = if child_index == internal.keys.len() {
          guard.high_key.as_ref()
        } else {
          Some(&internal.keys[child_index])
        };

        if !optional_keys_equal(child_guard.low_key.as_ref(), expected_low)
          || !optional_keys_equal(
            child_guard.high_key.as_ref(),
            expected_high,
          )
        {
          return Err(String::from(
            "child fences disagree with parent bounds",
          ));
        }
        drop(child_guard);

        validate_node(child, false, depth + 1, leaf_depth, record_count)?;
      }
    },
  }

  Ok(())
}

/// Checks every same-level forward/back link without allocating a `Vec`.
#[cfg(test)]
fn validate_sibling_levels<K: Ord, V, const N: usize>(
  root: &NodeRef<K, V, N>,
  len: usize,
  height: usize,
) -> Result<(), String> {
  let mut level_start = Arc::clone(root);
  let traversal_limit = len.saturating_add(height).saturating_add(2);

  for level in 0..height {
    let mut current = Arc::clone(&level_start);
    let mut previous: Option<NodeRef<K, V, N>> = None;
    let mut visited = 0;

    loop {
      visited += 1;
      if visited > traversal_limit {
        return Err(String::from("sibling chain contains a cycle"));
      }

      let guard = read_node(&current);
      if matches!(guard.kind, NodeKind::Redirect(_)) {
        return Err(String::from("sibling chain retains a redirect"));
      }

      match &previous {
        None => {
          if guard.low_key.is_some() || guard.previous.is_some() {
            return Err(String::from(
              "leftmost page has a predecessor fence",
            ));
          }
        },
        Some(previous_ref) => {
          let previous_guard = read_node(previous_ref);
          if !optional_keys_equal(
            previous_guard.high_key.as_ref(),
            guard.low_key.as_ref(),
          ) {
            return Err(String::from("adjacent sibling fences disagree"));
          }
          let back = guard
            .previous
            .as_ref()
            .and_then(Weak::upgrade)
            .ok_or_else(|| String::from("sibling lost its predecessor"))?;
          if !Arc::ptr_eq(&back, previous_ref) {
            return Err(String::from(
              "sibling predecessor points elsewhere",
            ));
          }
        },
      }

      match &guard.right {
        Some(next) => {
          let next = Arc::clone(next);
          drop(guard);
          previous = Some(current);
          current = next;
        },
        None => {
          if guard.high_key.is_some() {
            return Err(String::from(
              "rightmost page has a finite high fence",
            ));
          }
          break;
        },
      }
    }

    if level + 1 < height {
      let guard = read_node(&level_start);
      let child = match &guard.kind {
        NodeKind::Internal(internal) => Arc::clone(&internal.children[0]),
        NodeKind::Leaf(_) => {
          return Err(String::from("tree height exceeds leaf depth"));
        },
        NodeKind::Redirect(_) => unreachable!("validated above"),
      };
      drop(guard);
      level_start = child;
    }
  }

  Ok(())
}

/// Tests whether a key belongs to a half-open optional-fence interval.
#[cfg(test)]
fn within_fences<K: Ord>(
  key: &KeyHandle<K>,
  low: &Option<KeyHandle<K>>,
  high: &Option<KeyHandle<K>>,
) -> bool {
  low.as_ref().is_none_or(|bound| key >= bound)
    && high.as_ref().is_none_or(|bound| key < bound)
}

/// Compares optional key handles by their underlying key values.
#[cfg(test)]
fn optional_keys_equal<K: Ord>(
  left: Option<&KeyHandle<K>>,
  right: Option<&KeyHandle<K>>,
) -> bool {
  match (left, right) {
    (Some(left), Some(right)) => left == right,
    (None, None) => true,
    (Some(_), None) | (None, Some(_)) => false,
  }
}

#[cfg(test)]
mod tests {
  use std::{
    collections::BTreeMap,
    sync::{Arc, Barrier},
    thread,
  };

  use super::*;

  type TestTree = Tree<u64, u64, 3>;

  #[test]
  fn separator_equality_selects_the_right_child() {
    let mut keys = Array::with_capacity(4);
    keys.push(KeyHandle::new(10));
    keys.push(KeyHandle::new(20));

    let mut children = Array::with_capacity(5);
    children.push(Node::leaf_root());
    children.push(Node::leaf_root());
    children.push(Node::leaf_root());
    let internal = InternalNode::<u64, u64, 3> { keys, children };

    assert_eq!(internal.child_index(&5), 0);
    assert_eq!(internal.child_index(&10), 1);
    assert_eq!(internal.child_index(&15), 1);
    assert_eq!(internal.child_index(&20), 2);
  }

  #[test]
  fn ascending_insert_remove_reinsert_preserves_all_invariants() {
    let tree = TestTree::new();

    for key in 0..2_000 {
      assert_eq!(tree.insert(key, key * 2), None);
    }
    tree.validate().unwrap();

    for key in 0..2_000 {
      assert_eq!(tree.remove(&key), Some(key * 2));
      if key % 127 == 0 {
        tree.validate().unwrap();
      }
    }
    assert_eq!(tree.len(), 0);
    assert_eq!(tree.height(), 1);
    tree.validate().unwrap();

    for key in (0..2_000).rev() {
      tree.insert(key, key + 1);
    }
    tree.validate().unwrap();
  }

  #[test]
  fn differential_trace_matches_btree_map() {
    let tree = Tree::<u64, u64, 5>::new();
    let mut model = BTreeMap::new();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;

    for step in 0..20_000_u64 {
      state ^= state << 7;
      state ^= state >> 9;
      state ^= state << 8;
      let key = state % 500;

      match state % 4 {
        0 | 1 => {
          let value = state ^ step;
          assert_eq!(tree.insert(key, value), model.insert(key, value));
        },
        2 => assert_eq!(tree.remove(&key), model.remove(&key)),
        _ => {
          let actual = tree.get(&key).map(|(guard, index)| {
            let NodeKind::Leaf(leaf) = &guard.kind else {
              unreachable!();
            };
            leaf.entries[index].value
          });
          assert_eq!(actual, model.get(&key).copied());
        },
      }

      if step % 251 == 0 {
        tree.validate().unwrap();
        assert_eq!(tree.len(), model.len());
      }
    }

    tree.validate().unwrap();
  }

  #[test]
  fn concurrent_disjoint_writers_split_and_merge() {
    const THREADS: usize = 8;
    const PER_THREAD: usize = 1_000;

    let tree = Arc::new(Tree::<usize, usize, 7>::new());
    let start = Arc::new(Barrier::new(THREADS));
    let mut workers = Array::with_capacity(THREADS);

    for thread_index in 0..THREADS {
      let tree = Arc::clone(&tree);
      let start = Arc::clone(&start);
      workers.push(thread::spawn(move || {
        start.wait();
        for offset in 0..PER_THREAD {
          let key = thread_index * PER_THREAD + offset;
          assert_eq!(tree.insert(key, key), None);
        }
        for offset in (0..PER_THREAD).step_by(2) {
          let key = thread_index * PER_THREAD + offset;
          assert_eq!(tree.remove(&key), Some(key));
        }
      }));
    }

    while let Some(worker) = workers.pop() {
      worker.join().unwrap();
    }

    assert_eq!(tree.len(), THREADS * PER_THREAD / 2);
    tree.validate().unwrap();
    for key in 0..THREADS * PER_THREAD {
      assert_eq!(tree.get(&key).is_some(), key % 2 == 1);
    }
  }
}
