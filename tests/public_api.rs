use b_link::{
  BLinkMap, BLinkSet, EntryMut, EntryRef, Iter, Range, SetIter, SetRange,
  SetRef, DEFAULT_PAGE_CAPACITY,
};

fn accepts_read_guard(_entry: &EntryRef<'_, u64, String, 4>) {}

fn accepts_write_guard(_entry: &EntryMut<'_, u64, String, 4>) {}

fn accepts_default_read_guard(_entry: &EntryRef<'_, u64, String>) {}

fn accepts_default_write_guard(_entry: &EntryMut<'_, u64, String>) {}

#[test]
fn public_b_link_map_api_is_nameable() {
  let map = BLinkMap::<u64, String, 4>::new();

  assert_eq!(map.insert(7, String::from("seven")), None);

  let entry = map.get(&7).unwrap();
  accepts_read_guard(&entry);
  assert_eq!(entry.key(), &7);
  assert_eq!(entry.value(), "seven");
  drop(entry);

  let mut entry = map.get_mut(&7).unwrap();
  accepts_write_guard(&entry);
  entry.value_mut().push('!');
  drop(entry);

  assert_eq!(map.remove(&7), Some(String::from("seven!")));
  assert!(map.is_empty());
}

fn accepts_map_iterator(_iterator: Iter<'_, u64, String>) {}

fn accepts_map_range<R>(_range: Range<'_, u64, String, u64, R>)
where
  R: std::ops::RangeBounds<u64>,
{
}

fn accepts_set_ref(_value: &SetRef<'_, u64>) {}

fn accepts_set_iterator(_iterator: SetIter<'_, u64>) {}

fn accepts_set_range<R>(_range: SetRange<'_, u64, u64, R>)
where
  R: std::ops::RangeBounds<u64>,
{
}

#[test]
fn iterators_ranges_and_set_are_public() {
  let map = BLinkMap::<u64, String>::new();
  for key in 0..20 {
    map.insert(key, key.to_string());
  }

  accepts_map_iterator(map.iter());
  accepts_map_range(map.range(3..=8));

  let mut expected = 0;
  for entry in &map {
    assert_eq!(entry.key(), &expected);
    assert_eq!(entry.value(), &expected.to_string());
    expected += 1;
  }
  assert_eq!(expected, 20);

  let mut expected = 5;
  for entry in map.range(5..10) {
    assert_eq!(entry.key(), &expected);
    expected += 1;
  }
  assert_eq!(expected, 10);

  let set = BLinkSet::<u64>::new();
  assert!(set.insert(7));
  assert!(!set.insert(7));
  assert!(set.insert(11));
  assert!(set.contains(&7));

  let value = set.get(&7).unwrap();
  accepts_set_ref(&value);
  assert_eq!(*value, 7);
  drop(value);

  accepts_set_iterator(set.iter());
  accepts_set_range(set.range(7..=11));
  assert!(set.remove(&7));
  assert!(!set.remove(&7));
}

#[test]
fn default_capacity_and_unordered_empty_construction_are_public() {
  assert_eq!(DEFAULT_PAGE_CAPACITY, 32);

  let map = BLinkMap::<u64, String>::new();
  map.insert(1, String::from("one"));

  let entry = map.get(&1).unwrap();
  accepts_default_read_guard(&entry);
  drop(entry);

  let entry = map.get_mut(&1).unwrap();
  accepts_default_write_guard(&entry);
  drop(entry);

  struct NonOrd;

  let empty = BLinkMap::<NonOrd, ()>::new();
  assert_eq!(empty.len(), 0);
  assert!(empty.is_empty());
}
