//! Bloom alert model.
//!
//! The alert model is a simple atomic DEVS model that raises an alert when
//! any forecast horizon exceeds its threshold.
//! It receives forecasts  and checks them against
//! predefined thresholds for each horizon.
//! If a forecast value exceeds its corresponding threshold,
//! an alert is generated and sent to the output port.

use crate::forecast::Forecast;
use ubloom_model::OUT_DIM;
use xdevs::{Duration, prelude::*};

/// Raises an alert when any forecast horizon exceeds its threshold.
pub struct ThresholdAlert {
    alerts: [Option<bool>; OUT_DIM],
    threshold: f32,
    sigma: Duration,
}

impl ThresholdAlert {
    /// `threshold` is the same for all horizons.
    pub fn new(threshold: f32) -> Self {
        Self {
            alerts: [None; OUT_DIM],
            threshold,
            sigma: Duration::MAX,
        }
    }
}

impl Component for ThresholdAlert {
    type Kind = AtomicKind;
    type Input = Port<Forecast, 1>;
    type Output = [Port<bool, 1>; OUT_DIM];
}

impl Atomic for ThresholdAlert {
    fn delta_int(&mut self) {
        self.alerts = [None; OUT_DIM];
        self.sigma = Duration::MAX;
    }

    fn lambda(&self, output: &mut Self::Output) {
        for (&alert, output) in self.alerts.iter().zip(output.iter_mut()) {
            if let Some(alert) = alert {
                output.add_value(alert).unwrap();
            }
        }
    }

    fn ta(&self) -> Duration {
        self.sigma
    }

    fn delta_ext(&mut self, _: Duration, input: &Self::Input) {
        self.sigma = Duration::MAX;
        if let Some(&forecasts) = input.last() {
            for (&forecast, alert) in forecasts.iter().zip(self.alerts.iter_mut()) {
                // Check if the forecast exceeds the threshold
                if forecast > self.threshold {
                    *alert = Some(true);
                    self.sigma = Duration::ZERO; // an alert will be raised
                }
            }
        }
    }
}
