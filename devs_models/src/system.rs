//! Coupled DEVS model that wires the `Forecast` atomic model to the `Alert` atomic model.

use crate::{alert::*, forecast::*};
use ubloom_model::OUT_DIM;
use xdevs::prelude::*;

#[derive(xdevs::Bag)]
pub struct ForecastOutput {
    forecast: Port<Forecast, 1>,
    alerts: [Port<bool, 1>; OUT_DIM],
}

#[xdevs::coupled]
pub struct ForecastSystem {
    forecaster: Forecaster,
    alert: ThresholdAlert,
}

// RtEngine attributes change depending on the target.
#[cfg_attr(feature = "embassy", xdevs::rt_engine(max_out_subs = 2))]
#[cfg_attr(all(feature = "tokio", not(feature = "embassy")), xdevs::rt_engine)]
impl Component for ForecastSystem {
    type Kind = CoupledKind;
    type Input = Port<Window, 1>;
    type Output = ForecastOutput;
}

impl Coupled for ForecastSystem {
    fn eic(from: &Self::Input, to: &mut ComponentsInput<Self>) {
        couple(from, &mut to.forecaster).unwrap();
    }
    fn ic(from: &ComponentsOutput<Self>, to: &mut ComponentsInput<Self>) {
        couple(&from.forecaster, &mut to.alert).unwrap();
    }
    fn eoc(from: &ComponentsOutput<Self>, to: &mut Self::Output) {
        couple(&from.forecaster, &mut to.forecast).unwrap();
        couple(&from.alert, &mut to.alerts).unwrap();
    }
}
