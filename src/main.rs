#![windows_subsystem = "windows"]
mod app;
use app::SDUIApp;
use eframe::{egui, IconData};

fn main() {
    let ico = image::load_from_memory(include_bytes!("../assets/ico.png")).unwrap().to_rgba8();
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::Vec2::new(1200.0, 750.0)),
        initial_window_pos: Some(egui::Pos2::new(50.0, 50.0)),
        icon_data: Some(IconData{
            rgba: ico.into_raw(),
            width:256,
            height: 255,
        } ),
        ..Default::default()
    };
    eframe::run_native(
        "SDUI",
        options,
        Box::new(|_cc| Box::new(SDUIApp::new())),
    );
}
