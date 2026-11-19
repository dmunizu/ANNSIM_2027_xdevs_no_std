#![no_std]
#![no_main]
#![deny(
    clippy::mem_forget,
    reason = "mem::forget is generally not safe to do with esp_hal types, especially those \
    holding buffers for the duration of a data transfer."
)]
#![deny(clippy::large_stack_frames)]

use core::{fmt::Write as _, num::NonZero};

use devs_models::ForecastSystem;
use devs_models::system::{ForecastSystemOutputEnum, ForecastSystemReceiver};
use embassy_executor::Spawner;
use embassy_net::{IpAddress, IpEndpoint, Runner, Stack, tcp::TcpSocket};
use embassy_time::{Duration, Timer};
use esp_hal::{clock::CpuClock, rng::Rng, timer::timg::TimerGroup};
use esp_radio::wifi::{Config as WifiConfig, Interface, WifiController, sta::StationConfig};
use rust_mqtt::{
    Bytes,
    buffer::BumpBuffer,
    client::{
        Client,
        options::{ConnectOptions, PublicationOptions, TopicReference},
    },
    config::KeepAlive,
    types::{MqttString, TopicName},
};
use static_cell::StaticCell;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

extern crate alloc;

// This creates a default app-descriptor required by the esp-idf bootloader.
// For more information see: <https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-reference/system/app_image_format.html#application-description>
esp_bootloader_esp_idf::esp_app_desc!();

const SSID: &str = "my-ssid";
const PASSWORD: &str = "my-password";
const BROKER: IpEndpoint = IpEndpoint::new(IpAddress::v4(192, 168, 1, 10), 1883);
const CLIENT_ID: &str = "esp32";
const TOPIC: &str = "annsim_2027_xdevs/out";
// Keepalive offered in CONNECT; the broker disconnects after 1.5x this without traffic.
const KEEP_ALIVE_SECS: u16 = 60;
// PINGREQ interval, comfortably below the keepalive the broker enforces.
const PING_INTERVAL: Duration = Duration::from_secs(20);

/// Upper bound for the formatted forecast: one signed, 3-decimal value per horizon.
const PAYLOAD_LEN: usize = 64;

static NET_RESOURCES: StaticCell<embassy_net::StackResources<3>> = StaticCell::new();

#[embassy_executor::task]
async fn net_task(mut runner: Runner<'static, Interface>) {
    runner.run().await
}

#[embassy_executor::task]
async fn wifi_task(mut controller: WifiController<'static>) {
    loop {
        if controller.connect_async().await.is_ok() {
            controller.wait_for_disconnect_async().await.ok();
        } else {
            Timer::after(Duration::from_secs(1)).await;
        }
    }
}

/// Publishes every forecast produced by the DEVS model to the broker.
///
/// The MQTT session is owned by this task: on any transport or protocol error the
/// connection is dropped and re-established once the stack has a config again.
#[allow(
    clippy::large_stack_frames,
    reason = "the MQTT session plus its socket and encoder buffers live for the connection"
)]
#[embassy_executor::task]
async fn mqtt_task(stack: Stack<'static>, mut receiver: ForecastSystemReceiver) {
    let mut socket_rx = [0u8; 256];
    let mut socket_tx = [0u8; 256];

    loop {
        stack.wait_link_up().await;
        stack.wait_config_up().await;

        // A client owns its socket and session, so both are rebuilt per connection attempt.
        let mut mqtt_storage = [0u8; 256];
        let mut mqtt_buffer = BumpBuffer::new(&mut mqtt_storage);
        // Const generics: subscribe slots, receive maximum, send maximum, subscription ids,
        // user properties, incoming topic aliases, and outgoing topic aliases.
        let mut client = Client::<'_, '_, _, _, 1, 1, 1, 1, 1, 0, 0>::new(&mut mqtt_buffer);

        let mut socket = TcpSocket::new(stack, &mut socket_rx, &mut socket_tx);
        socket.set_timeout(Some(Duration::from_secs(10)));
        if socket.connect(BROKER).await.is_err() {
            Timer::after(Duration::from_secs(1)).await;
            continue;
        }

        let connect_options = ConnectOptions::new()
            .clean_start()
            .keep_alive(KeepAlive::Seconds(
                NonZero::new(KEEP_ALIVE_SECS).expect("non-zero keep alive"),
            ));
        let client_id = MqttString::from_str(CLIENT_ID).expect("valid client id");
        if client
            .connect(socket, &connect_options, Some(client_id))
            .await
            .is_err()
        {
            Timer::after(Duration::from_secs(1)).await;
            continue;
        }

        let topic = TopicName::new(MqttString::from_str(TOPIC).expect("valid topic"))
            .expect("valid topic name");

        loop {
            match embassy_time::with_timeout(PING_INTERVAL, receiver.recv()).await {
                // Lagged means the model outran the subscriber; the newest forecast follows.
                Ok(Err(_lagged)) => {}
                // Alerts are not part of the published payload.
                Ok(Ok(ForecastSystemOutputEnum::Alerts(_))) => {}
                Ok(Ok(ForecastSystemOutputEnum::Forecast(forecast))) => {
                    let mut payload = heapless::String::<PAYLOAD_LEN>::new();
                    let mut fits = true;
                    for (index, value) in forecast.iter().enumerate() {
                        if index > 0 {
                            fits &= payload.push(',').is_ok();
                        }
                        fits &= write!(payload, "{value}").is_ok();
                    }
                    if !fits {
                        continue;
                    }

                    // Reclaim the encoder workspace; the previous PUBLISH is already written.
                    unsafe { client.buffer_mut().reset() };
                    let publication =
                        PublicationOptions::new(TopicReference::Name(topic.as_borrowed()));
                    if client
                        .publish(&publication, Bytes::from(payload.as_bytes()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                // Nothing to publish: keep the session alive with a PINGREQ.
                Err(_) => {
                    if client.ping().await.is_err() {
                        break;
                    }
                }
            }

            if !stack.is_link_up() || !stack.is_config_up() {
                break;
            }
        }
    }
}

#[allow(
    clippy::large_stack_frames,
    reason = "it's not unusual to allocate larger buffers etc. in main"
)]
#[esp_rtos::main]
async fn main(spawner: Spawner) -> ! {
    // Initialize the ESP32 peripherals
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    esp_alloc::heap_allocator!(#[esp_hal::ram(reclaimed)] size: 65536);
    let timg0 = TimerGroup::new(peripherals.TIMG0);
    esp_rtos::start(timg0.timer0, peripherals.FROM_CPU_INTR0);

    // Wi-Fi setup
    let mut wifi_controller =
        esp_radio::wifi::WifiController::new(peripherals.WIFI, Default::default())
            .expect("Failed to initialize Wi-Fi controller");
    wifi_controller
        .set_config(&WifiConfig::Station(
            StationConfig::default()
                .with_ssid(SSID.try_into().expect("bad SSID"))
                .with_authentication(esp_radio::wifi::AuthenticationMethodConfig::Wpa2Personal(
                    PASSWORD.try_into().expect("bad password"),
                )),
        ))
        .expect("Failed to apply Wi-Fi config");

    let rng = Rng::new();
    let seed = (rng.random() as u64) << 32 | rng.random() as u64;

    // The station interface is a singleton: it can be created once and then owned by the stack.
    let (stack, runner) = embassy_net::new(
        Interface::station(),
        embassy_net::Config::dhcpv4(Default::default()),
        NET_RESOURCES.init(embassy_net::StackResources::new()),
        seed,
    );

    // DEVS model setup
    let forecast_system = ForecastSystem::new(0.5);
    let mut rt_engine = forecast_system.into_rt_engine();
    let sim_config = xdevs::RtConfig::<xdevs::export::Clock>::default();

    // Subscribe before the engine runs.
    let receiver = rt_engine
        .receiver()
        .expect("output channel subscriber slot");

    // Spawn the network, Wi-Fi, and MQTT tasks. The DEVS simulation runs in the main task.
    spawner.spawn(net_task(runner).expect("failed to spawn network task"));
    spawner.spawn(wifi_task(wifi_controller).expect("failed to spawn Wi-Fi task"));
    spawner.spawn(mqtt_task(stack, receiver).expect("failed to spawn MQTT task"));

    // Run the DEVS simulation.
    rt_engine
        .simulate_rt(xdevs::Duration::MAX, sim_config)
        .await;

    loop {
        Timer::after(Duration::from_secs(1)).await;
    }
}
