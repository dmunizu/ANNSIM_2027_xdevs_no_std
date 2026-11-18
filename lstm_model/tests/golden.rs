//! Bit-exactness against 100 golden vectors.
//!
//! The expected outputs are stored as raw f32 bit patterns, so this compares
//! exactly.

#[path = "data/golden_data.rs"]
#[rustfmt::skip]
mod golden_data;

use ubloom_model::{FEATURES, OUT_DIM, predict};

#[test]
fn golden_vectors_are_bit_exact() {
    let mut failures = Vec::new();

    for (i, (features, expected)) in golden_data::VECTORS.iter().enumerate() {
        let out = predict(features);
        let bits: [u32; OUT_DIM] = core::array::from_fn(|k| out[k].to_bits());
        if &bits != expected {
            failures.push((i, bits, *expected));
        }
    }

    if !failures.is_empty() {
        let mut msg = format!(
            "{} of {} vectors differ from the golden output\n",
            failures.len(),
            golden_data::VECTORS.len()
        );
        for (i, got, want) in failures.iter().take(5) {
            msg += &format!("  vector {i}:\n    got  {got:08X?}\n    want {want:08X?}\n");
        }
        panic!("{msg}");
    }
}

#[test]
fn feature_length_is_what_the_model_expects() {
    // 30 timesteps of 3 sensors, which is what the firmware sends.
    assert_eq!(FEATURES, 90);
    assert_eq!(
        golden_data::VECTORS.len(),
        100,
        "expected the full golden set"
    );
    assert_eq!(golden_data::VECTORS[0].0.len(), FEATURES);
}
