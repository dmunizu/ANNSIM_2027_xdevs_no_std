//! 30-step recurrence of the LSTM model.
//!
//! Two things to take into account when comparing this to a textbook LSTM:
//!
//!   * There is no tanh anywhere in this model, using ReLU instead.
//!   * The last timestep multiplies the output gate by the cell state
//!     directly, with no ReLU in between.
//!
//! Each function names the C code it mirrors from esp-nn or TFLM.

use crate::{data::*, fixed::*};
use core::cmp::{max, min};

/// `esp_nn_fully_connected_s8`.
///
/// Affine map `out = weight * input + bias`.
///
/// The filter offset is always 0 here (TFLM requires symmetric int8
/// weights).
fn fc(
    input: &[i8],
    weights: &[i8],
    acc_depth: usize,
    bias: Option<&[i32]>,
    params: &Fc,
    out: &mut [i8],
) {
    for (oc, slot) in out.iter_mut().enumerate() {
        let base = oc * acc_depth;
        let mut acc: i32 = 0;
        for i in 0..acc_depth {
            acc += (weights[base + i] as i32) * (input[i] as i32 - params.zp_in);
        }
        if let Some(b) = bias {
            acc += b[oc];
        }
        let v = multiply_by_quantized_mult(acc, params.req.mult, params.req.shift) + params.zp_out;
        *slot = v.clamp(-128, 127) as i8;
    }
}

/// `esp_nn_add_elementwise_s8`
///
/// Adds two tensors that use different scales: each is rescaled to the output
/// scale first, then the sum is rescaled once more.
fn add(a: &[i8], b: &[i8], params: &Add, out: &mut [i8]) {
    for i in 0..a.len() {
        let t1 = sat_round_doubling_high_mul(
            (a[i] as i32 - params.zp1) << (params.left_shift as u32),
            params.in1.mult,
        );
        let t2 = sat_round_doubling_high_mul(
            (b[i] as i32 - params.zp2) << (params.left_shift as u32),
            params.in2.mult,
        );
        // We use max for negative values to clamp to 0 instead of turning into a bit-equivalent u32.
        let t1 = div_by_power_of_two(t1, max(-params.in1.shift, 0) as u32);
        let t2 = div_by_power_of_two(t2, max(-params.in2.shift, 0) as u32);

        let pre = sat_round_doubling_high_mul(t1 + t2, params.out.mult);
        let v = div_by_power_of_two(pre, max(-params.out.shift, 0) as u32);
        out[i] = (v + params.zp_out).clamp(-128, 127) as i8;
    }
}

/// `esp_nn_mul_elementwise_s8`.
///
/// Elementwise product rescaled to the output scale. The gate times the thing it gates.
fn mul(a: &[i8], b: &[i8], params: &Mul, out: &mut [i8]) {
    for i in 0..a.len() {
        let product = (a[i] as i32 - params.zp1) * (b[i] as i32 - params.zp2);
        let v =
            multiply_by_quantized_mult(product, params.req.mult, params.req.shift) + params.zp_out;
        out[i] = v.clamp(-128, 127) as i8;
    }
}

/// TFLM `ReluQuantized`.
///
/// Change the scale of the input to the output scale, clamping to the output range.
fn relu(x: &[i8], params: &Relu, out: &mut [i8]) {
    for i in 0..x.len() {
        let v = multiply_by_quantized_mult(
            x[i] as i32 - params.zp_in,
            params.req.mult,
            params.req.shift,
        ) + params.zp_out;
        out[i] = min(params.max, max(params.min, v)) as i8;
    }
}

/// TFLM `Logistic`.
///
/// A 256-entry gemmlowp table read at the input
/// code point. It is a sigmoid through lookup of a pre-computed table, not arithmetic.
#[inline]
fn logistic(x: &[i8], table_idx: u8, out: &mut [i8]) {
    let t = &LOGISTIC[table_idx as usize];
    for i in 0..x.len() {
        out[i] = t[(x[i] as i32 + 128) as usize];
    }
}

/// One inference is composed of TIMESTEPS timesteps of IN_PER_STEP features each, giving OUT_DIM forecast horizons.
pub fn predict(features: &[i8; TIMESTEPS * IN_PER_STEP]) -> [f32; OUT_DIM] {
    // Both hidden and cell states start at the zero vector.
    // Preamble FILL produced a zero vector, so there is no learned initial state.
    let mut h = [0i8; HIDDEN];
    let mut c = [0i8; HIDDEN];

    // Intermediate buffers for the gates and their contributions to the cell state.
    let mut z_x = [0i8; GATES];
    let mut z_h = [0i8; GATES];
    let mut z_sum = [0i8; GATES];
    let mut z = [0i8; GATES];
    let mut i_t = [0i8; HIDDEN];
    let mut f_t = [0i8; HIDDEN];
    let mut o_t = [0i8; HIDDEN];
    let mut g_t = [0i8; HIDDEN];
    let mut input_contrib = [0i8; HIDDEN];
    let mut forget_contrib = [0i8; HIDDEN];
    let mut c_new = [0i8; HIDDEN];
    let mut c_relu = [0i8; HIDDEN];
    let mut h_new = [0i8; HIDDEN];

    for (t, params) in STEPS.iter().enumerate() {
        let base = t * IN_PER_STEP;
        let x = &features[base..base + IN_PER_STEP];

        // Gate pre-activations: z = FC(x_t) + FC(h) + bias.
        fc(x, &W_X, IN_PER_STEP, None, &params.fc_x, &mut z_x);
        fc(&h, &W_H, HIDDEN, None, &params.fc_h, &mut z_h);
        add(&z_x, &z_h, &params.add_z, &mut z_sum);
        add(&z_sum, &B_GATE, &params.add_b, &mut z);

        // Split into i, f, g, o: input, forget, candidate and output gates.
        let (gate_i, rest) = z.split_at(HIDDEN);
        let (gate_f, rest) = rest.split_at(HIDDEN);
        let (gate_g, gate_o) = rest.split_at(HIDDEN);

        relu(gate_g, &params.relu_g, &mut g_t);
        logistic(gate_i, params.log_i, &mut i_t);
        logistic(gate_f, params.log_f, &mut f_t);
        logistic(gate_o, params.log_o, &mut o_t);

        // Cell state: c_t = f * c_{t-1} + i * g, elementwise.
        mul(&i_t, &g_t, &params.mul_ig, &mut input_contrib);
        mul(&f_t, &c, &params.mul_fc, &mut forget_contrib);
        add(&forget_contrib, &input_contrib, &params.add_c, &mut c_new);

        if params.relu_cell_state {
            relu(&c_new, &params.relu_c, &mut c_relu);
            mul(&o_t, &c_relu, &params.mul_h, &mut h_new);
        } else {
            // The final timestep: o * c, no ReLU.
            mul(&o_t, &c_new, &params.mul_h, &mut h_new);
        }

        h = h_new;
        c = c_new;
    }

    // Output head: two projections, the second one with an int32 bias, then a
    // dequantise. This is the only floating point in the model.
    let mut head = [0i8; HIDDEN];
    fc(&h, &W_HEAD, HIDDEN, None, &HEAD_FC, &mut head);
    let mut q_out = [0i8; OUT_DIM];
    fc(&head, &W_OUT, HIDDEN, Some(&B_OUT), &OUT_FC, &mut q_out);

    let mut out = [0f32; OUT_DIM];
    for i in 0..OUT_DIM {
        out[i] = ((q_out[i] as i32 - OUT_ZP) as f32) * OUT_SCALE;
    }
    out
}
