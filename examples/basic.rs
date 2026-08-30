use b_link::{BLinkMap, BLinkSet};

fn main() {
  let map = BLinkMap::<u64, String>::new();
  map.insert(7, String::from("seven"));

  let set = BLinkSet::<u64>::new();
  set.insert(7);

  let value = map.get(&7).expect("the example inserted key 7");

  println!(
    "map[7] = {}, set contains 7 = {}",
    value.value(),
    set.contains(&7)
  );
}
