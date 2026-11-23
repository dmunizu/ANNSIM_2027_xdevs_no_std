//! Bloom alert model.
//!
//! The alert model is a simple atomic DEVS model that reports when a forecast
//! horizon crosses its threshold. It receives forecasts, compares every horizon
//! against the threshold, and emits `true` on the transition into alert and
//! `false` on the transition out of it. A horizon that stays on the same side of
//! the threshold produces no output.

use crate::forecast::Forecast;
use ubloom_model::OUT_DIM;
use xdevs::{Duration, prelude::*};

/// Reports threshold crossings of the forecast horizons.
pub struct ThresholdAlert {
    /// Outputs queued for the next `lambda`.
    alerts: [Option<bool>; OUT_DIM],
    /// Level of every horizon as of the last processed forecast.
    level: [bool; OUT_DIM],
    threshold: f32,
    sigma: Duration,
}

impl ThresholdAlert {
    /// `threshold` is the same for all horizons.
    pub fn new(threshold: f32) -> Self {
        Self {
            alerts: [None; OUT_DIM],
            level: [false; OUT_DIM],
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
            for ((&forecast, level), alert) in forecasts
                .iter()
                .zip(self.level.iter_mut())
                .zip(self.alerts.iter_mut())
            {
                let raised = forecast > self.threshold;
                if raised != *level {
                    *level = raised;
                    *alert = Some(raised);
                    self.sigma = Duration::ZERO; // send alert on the next `lambda`
                }
            }
        }
    }
}
