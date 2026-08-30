use std::{
  alloc::{alloc, dealloc, handle_alloc_error, Layout},
  fmt,
  marker::PhantomData,
  mem,
  ops::{Index, IndexMut},
  ptr::{self, NonNull},
  slice,
};

/// A fixed-capacity, contiguous, owning array.
///
/// The allocation is created once and never moved or grown. Tree pages split
/// or merge before a transformation could exceed their stable capacity.
/// Unlike a general-purpose growable collection, exceeding the declared
/// capacity is always a logic error and panics.
///
/// # Representation invariants
///
/// - `ptr` is non-null and aligned for `T`.
/// - For non-zero-sized `T`, `ptr` owns an allocation described by
///   `Layout::array::<T>(cap)`.
/// - Exactly the range `0..len` contains initialized values.
/// - `len <= cap`, and `cap` is always non-zero.
pub(crate) struct Array<T> {
  /// Start of the allocation, or an aligned dangling pointer for a ZST.
  ptr: NonNull<T>,
  /// Number of initialized elements beginning at `ptr`.
  len: usize,
  /// Maximum number of elements the allocation can hold.
  cap: usize,
  /// Expresses ownership of `T` for variance, drop checking, and auto traits.
  _marker: PhantomData<T>,
}

// Keep the complete fixed-capacity collection surface available even though
// the tree currently exercises only a subset of it in non-test builds.
#[allow(dead_code)]
impl<T> Array<T> {
  /// Allocates storage for exactly `cap` values without initializing any.
  ///
  /// # Panics
  ///
  /// Panics if `cap` is zero or if the requested allocation layout cannot be
  /// represented. Allocation failure invokes the standard allocation-error
  /// handler.
  pub fn with_capacity(cap: usize) -> Self {
    assert!(cap > 0, "capacity must be greater than zero");

    let layout = Layout::array::<T>(cap).expect("Array capacity overflow");

    let ptr = if mem::size_of::<T>() == 0 {
      // ZST values occupy no bytes, but references still require a non-null,
      // correctly aligned pointer.
      NonNull::dangling()
    } else {
      // SAFETY: `layout` has non-zero size and a valid alignment for `T`.
      let raw = unsafe { alloc(layout) };

      NonNull::new(raw.cast()).unwrap_or_else(|| handle_alloc_error(layout))
    };

    Self {
      ptr,
      len: 0,
      cap,
      _marker: PhantomData,
    }
  }

  /// Returns the number of initialized elements.
  pub fn len(&self) -> usize {
    self.len
  }

  /// Returns the fixed maximum number of elements.
  pub fn capacity(&self) -> usize {
    self.cap
  }

  /// Returns true when no elements are initialized.
  pub fn is_empty(&self) -> bool {
    self.len == 0
  }

  /// Returns true when another insertion would exceed the fixed capacity.
  pub fn is_full(&self) -> bool {
    self.len == self.cap
  }

  /// Views all initialized elements as an immutable slice.
  pub fn as_slice(&self) -> &[T] {
    // SAFETY:
    // `ptr` is aligned for `T`, and exactly the first `len` elements are
    // initialized for the lifetime of this borrow.
    unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
  }

  /// Views all initialized elements as a mutable slice.
  pub fn as_mut_slice(&mut self) -> &mut [T] {
    // SAFETY:
    // The mutable borrow of the array is exclusive, `ptr` is aligned for
    // `T`, and exactly the first `len` elements are initialized.
    unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
  }

  /// Returns the first initialized element, if one exists.
  pub fn first(&self) -> Option<&T> {
    self.as_slice().first()
  }

  /// Returns the last initialized element, if one exists.
  pub fn last(&self) -> Option<&T> {
    self.as_slice().last()
  }

  /// Moves the range `at..len` into a new array with the same capacity.
  ///
  /// No `T` value is cloned. Ownership of each moved element transfers to the
  /// returned array, while `self` retains `0..at`.
  ///
  /// # Panics
  ///
  /// Panics if `at > len` or if allocating the destination array fails.
  pub fn split_off(&mut self, at: usize) -> Self {
    assert!(at <= self.len, "index out of bounds");

    let old_len = self.len;

    let mut other = Self::with_capacity(self.cap);

    // SAFETY:
    // - `at <= old_len <= cap`, so the source range is initialized.
    // - `other` is a distinct allocation with capacity `cap`.
    // - The copied ranges cannot overlap and both pointers are aligned.
    unsafe {
      let src = self.ptr.as_ptr().add(at);
      let dst = other.ptr.as_ptr();

      ptr::copy_nonoverlapping(src, dst, old_len - at);
    }

    // Change ownership only after the infallible raw copy. The source's moved
    // tail is now outside its initialized range and must not be dropped there.
    other.len = old_len - at;
    self.len = at;

    other
  }

  /// Moves every element from `other` onto the end of `self`.
  ///
  /// `other` is empty afterward. Both arrays keep their original allocations.
  ///
  /// # Panics
  ///
  /// Panics if the combined length overflows or exceeds `self.capacity()`.
  pub fn append_from(&mut self, other: &mut Self) {
    let new_len = self
      .len
      .checked_add(other.len)
      .expect("Array length overflow");

    assert!(new_len <= self.cap, "Array capacity exceeded");

    if other.len == 0 {
      return;
    }

    // SAFETY:
    // - The capacity check proves the destination tail has enough space.
    // - `other` and `self` are distinct mutable borrows, hence disjoint arrays.
    // - `0..other.len` is initialized and the destination tail is not.
    unsafe {
      ptr::copy_nonoverlapping(
        other.ptr.as_ptr(),
        self.ptr.as_ptr().add(self.len),
        other.len,
      );
    }

    self.len = new_len;
    // The copied values are now owned by `self`; suppress their old drops.
    other.len = 0;
  }

  /// Moves every element from `other` onto the beginning of `self`.
  ///
  /// Existing elements are shifted to the right without cloning them, and
  /// `other` is empty afterward. This operation is used when deletion merges
  /// a left B-tree page into its surviving right sibling.
  ///
  /// # Panics
  ///
  /// Panics if the combined length overflows or exceeds `self.capacity()`.
  pub fn prepend_from(&mut self, other: &mut Self) {
    let prefix_len = other.len;
    let new_len = self
      .len
      .checked_add(prefix_len)
      .expect("Array length overflow");

    assert!(new_len <= self.cap, "Array capacity exceeded");

    if prefix_len == 0 {
      return;
    }

    // SAFETY:
    // - The capacity check leaves `prefix_len` free slots at the end.
    // - `ptr::copy` permits overlap while shifting `self` to the right.
    // - `self` and `other` are distinct mutable borrows, so the subsequent
    //   ownership transfer uses non-overlapping allocations.
    unsafe {
      ptr::copy(
        self.ptr.as_ptr(),
        self.ptr.as_ptr().add(prefix_len),
        self.len,
      );
      ptr::copy_nonoverlapping(
        other.ptr.as_ptr(),
        self.ptr.as_ptr(),
        prefix_len,
      );
    }

    self.len = new_len;
    // The prefix now belongs to `self`; exclude it from `other`'s drop range.
    other.len = 0;
  }

  /// Drops every initialized element and leaves the allocation reusable.
  ///
  /// Elements are removed from the initialized range before their destructors
  /// run, so catching a destructor panic cannot expose or double-drop a value
  /// whose destructor has already started.
  pub fn clear(&mut self) {
    // Reduce the initialized range before invoking user drop glue. If a
    // destructor unwinds, the array remains valid and cannot drop that element
    // a second time when the caller catches the panic.
    while self.len != 0 {
      self.len -= 1;

      // SAFETY: the decremented `len` was the index of the last initialized
      // element. It is removed from the logical range before drop glue runs.
      unsafe {
        ptr::drop_in_place(self.ptr.as_ptr().add(self.len));
      }
    }
  }

  /// Appends `value` to the initialized range.
  ///
  /// # Panics
  ///
  /// Panics when the array is full.
  pub fn push(&mut self, value: T) {
    assert!(self.len < self.cap, "Array capacity exceeded");

    // SAFETY: `len < cap`, so `ptr.add(len)` is an aligned, uninitialized slot
    // inside the allocation.
    unsafe {
      let dst = self.ptr.as_ptr().add(self.len);
      ptr::write(dst, value);
    }

    self.len += 1;
  }

  /// Inserts `value` at `index`, shifting the initialized suffix right.
  ///
  /// # Panics
  ///
  /// Panics if `index > len` or if the array is full.
  pub fn insert(&mut self, index: usize, value: T) {
    assert!(index <= self.len, "index out of bounds");
    assert!(self.len < self.cap, "Array capacity exceeded");

    // SAFETY:
    // - The assertions establish `index <= len < cap`.
    // - `ptr::copy` permits the overlapping one-slot right shift.
    // - After the shift, `index` is written without dropping the stale bytes
    //   there because ownership of that value moved to `index + 1`.
    unsafe {
      let base = self.ptr.as_ptr();

      ptr::copy(base.add(index), base.add(index + 1), self.len - index);
      ptr::write(base.add(index), value);
    }

    self.len += 1;
  }

  /// Removes and returns the element at `index`, shifting its suffix left.
  ///
  /// # Panics
  ///
  /// Panics if `index >= len`.
  pub fn remove(&mut self, index: usize) -> T {
    assert!(index < self.len, "index out of bounds");

    // SAFETY:
    // - `index < len`, so reading the removed value is valid.
    // - `ptr::copy` permits the overlapping left shift.
    // - The stale bytes at the old last slot fall outside the decremented
    //   initialized range and must not be dropped a second time.
    unsafe {
      let base = self.ptr.as_ptr();
      let value = ptr::read(base.add(index));

      ptr::copy(base.add(index + 1), base.add(index), self.len - index - 1);

      self.len -= 1;

      value
    }
  }

  /// Removes and returns the last element, or `None` when empty.
  pub fn pop(&mut self) -> Option<T> {
    if self.len == 0 {
      return None;
    }

    // Remove the slot from the initialized range before moving its value out.
    self.len -= 1;

    // SAFETY: the old last index is initialized and is now outside `0..len`.
    Some(unsafe { ptr::read(self.ptr.as_ptr().add(self.len)) })
  }

  /// Returns the element at `index`, or `None` when out of bounds.
  pub fn get(&self, index: usize) -> Option<&T> {
    if index < self.len {
      // SAFETY:
      // index < len, and every element in [0, len) is initialized.
      Some(unsafe { self.get_unchecked(index) })
    } else {
      None
    }
  }

  /// Returns the element at `index` mutably, or `None` when out of bounds.
  pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
    if index < self.len {
      // SAFETY:
      // index < len, and every element in [0, len) is initialized.
      Some(unsafe { self.get_unchecked_mut(index) })
    } else {
      None
    }
  }

  /// Returns an element without checking the index.
  ///
  /// # Safety
  ///
  /// `index` must be less than [`Array::len`].
  pub unsafe fn get_unchecked(&self, index: usize) -> &T {
    debug_assert!(index < self.len);

    // SAFETY: the caller guarantees `index < len`; the representation
    // invariant says that range is initialized and aligned.
    unsafe { &*self.ptr.as_ptr().add(index) }
  }

  /// Returns an element mutably without checking the index.
  ///
  /// # Safety
  ///
  /// `index` must be less than [`Array::len`].
  pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
    debug_assert!(index < self.len);

    // SAFETY: the caller guarantees `index < len`, and `&mut self` proves this
    // returned reference is exclusive for its borrow duration.
    unsafe { &mut *self.ptr.as_ptr().add(index) }
  }
}

impl<T> Index<usize> for Array<T> {
  type Output = T;

  fn index(&self, index: usize) -> &T {
    &self.as_slice()[index]
  }
}

impl<T> IndexMut<usize> for Array<T> {
  fn index_mut(&mut self, index: usize) -> &mut T {
    &mut self.as_mut_slice()[index]
  }
}

impl<T: fmt::Debug> fmt::Debug for Array<T> {
  fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
    self.as_slice().fmt(formatter)
  }
}

impl<T> Drop for Array<T> {
  fn drop(&mut self) {
    // Drop values before releasing the allocation they occupy.
    self.clear();

    if mem::size_of::<T>() != 0 {
      let layout = Layout::array::<T>(self.cap)
        .expect("an Array must retain its original valid layout");

      // SAFETY: construction allocated `ptr` using this exact layout, and
      // `clear` has removed every initialized value.
      unsafe { dealloc(self.ptr.as_ptr().cast(), layout) }
    }
    // ZST arrays use an aligned dangling pointer and have no allocation to
    // release.
  }
}

// SAFETY: `Array` owns its elements and allocation. Moving it between threads
// is sound exactly when moving `T` is sound.
unsafe impl<T: Send> Send for Array<T> {}

// SAFETY: shared access exposes only `&T`; mutation requires `&mut self`.
// Therefore sharing an `Array<T>` is sound exactly when sharing `T` is sound.
unsafe impl<T: Sync> Sync for Array<T> {}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn push_and_get() {
    let mut array = Array::with_capacity(4);

    array.push(10);
    array.push(20);
    array.push(30);

    assert_eq!(array.len(), 3);
    assert_eq!(array.capacity(), 4);

    assert_eq!(array.get(0), Some(&10));
    assert_eq!(array.get(1), Some(&20));
    assert_eq!(array.get(2), Some(&30));
    assert_eq!(array.get(3), None);
  }

  #[test]
  fn insert_middle() {
    let mut array = Array::with_capacity(8);

    array.push(10);
    array.push(20);
    array.push(40);

    array.insert(2, 30);

    assert_eq!(array.as_slice(), &[10, 20, 30, 40]);
  }

  #[test]
  fn remove_middle() {
    let mut array = Array::with_capacity(8);

    array.push(10);
    array.push(20);
    array.push(30);
    array.push(40);

    let removed = array.remove(1);

    assert_eq!(removed, 20);

    assert_eq!(array.as_slice(), &[10, 30, 40]);
  }

  #[test]
  fn pop() {
    let mut array = Array::with_capacity(4);

    array.push(10);
    array.push(20);

    assert_eq!(array.pop(), Some(20));
    assert_eq!(array.pop(), Some(10));
    assert_eq!(array.pop(), None);
  }

  #[test]
  fn drop_values() {
    use std::cell::Cell;
    use std::rc::Rc;

    struct Droppable {
      counter: Rc<Cell<usize>>,
    }

    impl Drop for Droppable {
      fn drop(&mut self) {
        self.counter.set(self.counter.get() + 1);
      }
    }

    let counter = Rc::new(Cell::new(0));

    {
      let mut array = Array::with_capacity(8);

      array.push(Droppable {
        counter: Rc::clone(&counter),
      });

      array.push(Droppable {
        counter: Rc::clone(&counter),
      });

      let _ = array.remove(0);

      assert_eq!(counter.get(), 1);
    }

    assert_eq!(counter.get(), 2);
  }

  #[test]
  fn clear_remains_valid_when_an_element_destructor_panics() {
    use std::{
      cell::{Cell, RefCell},
      panic::{catch_unwind, AssertUnwindSafe},
      rc::Rc,
    };

    struct DropBomb {
      id: usize,
      drops: Rc<RefCell<[usize; 3]>>,
      armed: Rc<Cell<bool>>,
    }

    impl Drop for DropBomb {
      fn drop(&mut self) {
        self.drops.borrow_mut()[self.id] += 1;

        if self.id == 1 && self.armed.replace(false) {
          panic!("intentional destructor panic");
        }
      }
    }

    let drops = Rc::new(RefCell::new([0; 3]));
    let armed = Rc::new(Cell::new(true));
    let mut array = Array::with_capacity(3);

    for id in 0..3 {
      array.push(DropBomb {
        id,
        drops: Rc::clone(&drops),
        armed: Rc::clone(&armed),
      });
    }

    let result = catch_unwind(AssertUnwindSafe(|| array.clear()));
    assert!(result.is_err());

    // Values 2 and 1 were removed from the initialized range before their
    // destructors ran. Only value 0 remains owned by the array.
    assert_eq!(array.len(), 1);
    assert_eq!(*drops.borrow(), [0, 1, 1]);

    array.clear();
    assert_eq!(*drops.borrow(), [1, 1, 1]);
  }

  #[test]
  fn split_off_transfers_ownership() {
    let mut left = Array::with_capacity(8);

    for value in 0..6 {
      left.push(value);
    }

    let mut right = left.split_off(2);

    assert_eq!(left.as_slice(), &[0, 1]);
    assert_eq!(right.as_slice(), &[2, 3, 4, 5]);

    left.append_from(&mut right);

    assert_eq!(left.as_slice(), &[0, 1, 2, 3, 4, 5]);
    assert!(right.is_empty());
  }

  #[test]
  fn supports_zero_sized_values() {
    let mut array = Array::with_capacity(3);

    array.push(());
    array.insert(0, ());
    array.push(());

    assert_eq!(array.len(), 3);
    assert!(array.is_full());
    assert_eq!(array.remove(1), ());
    assert_eq!(array.len(), 2);
  }
}
