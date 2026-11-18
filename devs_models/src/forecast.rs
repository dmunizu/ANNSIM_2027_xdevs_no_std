//! LSTM-based bloom forecaster.
//!
//! One event in, one forecast out. The input port carries a complete feature
//! window: 30 blocks x 3 sensors from the 120 days of data.
//! The model answers with four forecast horizons.

use ubloom_model::{FEATURES, OUT_DIM, predict};
use xdevs::{Duration, prelude::*};

/// One complete model input, what `predict` consumes.
pub type Window = [i8; FEATURES];

/// LSTM output: one forecast value for every 4-day horizon.
pub type Forecast = [f32; OUT_DIM];

/// Applies the LSTM to each window it receives.
///
/// Passive until the first window arrives. Each window produces one forecast.
pub struct Forecaster {
    forecast: Forecast,
    sigma: Duration,
}

impl Forecaster {
    pub fn new() -> Self {
        Self {
            forecast: [0.0; OUT_DIM],
            sigma: Duration::MAX,
        }
    }
}

impl Default for Forecaster {
    fn default() -> Self {
        Self::new()
    }
}

impl Component for Forecaster {
    type Kind = AtomicKind;
    type Input = Port<Window, 1>;
    type Output = Port<Forecast, 1>;
}

impl Atomic for Forecaster {
    fn delta_int(&mut self) {
        self.sigma = Duration::MAX;
    }

    fn lambda(&self, output: &mut Self::Output) {
        output.add_value(self.forecast).unwrap();
    }

    fn ta(&self) -> Duration {
        self.sigma
    }

    fn delta_ext(&mut self, elapsed: Duration, input: &Self::Input) {
        self.sigma -= elapsed;
        if let Some(&window) = input.last() {
            self.forecast = predict(&window);
            self.sigma = Duration::ZERO;
        }
    }
}
