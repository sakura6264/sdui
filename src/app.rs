use eframe::{egui, emath::Numeric};
use pyke_diffusers::{
    DDIMScheduler, DiffusionDevice, DiffusionDeviceControl, OrtEnvironment, Prompt,
    SchedulerOptimizedDefaults, StableDiffusionMemoryOptimizedPipeline, StableDiffusionOptions,
    StableDiffusionPipeline, StableDiffusionTxt2ImgOptions,
};
use std::io::{Cursor, Write};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread;
pub struct SDUIApp {
    // Main Window
    positive_prompt: String,
    negative_prompt: String,
    iters: usize,
    width: u32,
    height: u32,
    seed: u64,
    image: Option<egui_extras::RetainedImage>,
    rawpng: Option<Vec<u8>>,
    recvchannel: Option<Receiver<Vec<u8>>>, // channel to receive image data
    hthread: Option<std::thread::JoinHandle<()>>,
    timer: std::time::Instant,
    no_img: egui_extras::RetainedImage,
    use_dml: bool,
    use_low_mem: bool,
}

impl SDUIApp {
    pub fn new() -> Self {
        Self {
            positive_prompt: String::from(""),
            negative_prompt: String::from(""),
            iters: 32,
            width: 512,
            height: 512,
            image: None,
            seed: rand::random(),
            rawpng: None,
            recvchannel: None,
            hthread: None,
            timer: std::time::Instant::now(),
            no_img: egui_extras::RetainedImage::from_image_bytes(
                "no img",
                include_bytes!("../assets/no_image.png"),
            )
            .unwrap(),
            use_dml: true,
            use_low_mem: false,
        }
    }
}

impl eframe::App for SDUIApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // check if the num is divisible by 8
        if self.width % 8 != 0 {
            self.width = (self.width / 8) * 8;
        }
        if self.height % 8 != 0 {
            self.height = (self.height / 8) * 8;
        }
        //message from buttons
        let mut g_bmsg = false;
        let mut s_bmsg = false;
        let mut s_bmsgnf = false;
        let mut p_bmsg = false;
        let mut n_bmsg = false;
        let mut s_err: Option<std::io::Error> = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    egui::widgets::global_dark_light_mode_buttons(ui);
                    ui.heading("Stable Diffusion");
                    // Add UI elements here
                    // the UI has two text editor to edit prompts
                    // and two dragvalue to set the width and height of the image
                    // and a dragvalue to set the number of iterations
                    // and a button to generate the image
                    // and a button to save the image
                    // and a dragvalue to edit seed
                    ui.horizontal(|ui| {
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label("Positive Prompt: ");
                                p_bmsg = ui.button("Load TXT").clicked();
                                if ui.button("Clear").clicked() {
                                    self.positive_prompt.clear();
                                }
                            });
                            ui.add(
                                egui::widgets::text_edit::TextEdit::multiline(
                                    &mut self.positive_prompt,
                                )
                                .desired_rows(20)
                                .desired_width(200.0),
                            );
                        });
                        ui.separator();
                        ui.vertical(|ui| {
                            ui.horizontal(|ui| {
                                ui.label("Negative Prompt: ");
                                n_bmsg = ui.button("Load TXT").clicked();
                                if ui.button("Clear").clicked() {
                                    self.negative_prompt.clear();
                                }
                            });
                            ui.add(
                                egui::widgets::text_edit::TextEdit::multiline(
                                    &mut self.negative_prompt,
                                )
                                .desired_rows(20)
                                .desired_width(200.0),
                            );
                        });
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.label("Width: ");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.width)
                                .speed(8.0)
                                .clamp_range(8..=4096),
                        )
                        .on_hover_text("Must be divisible by 8");
                        ui.label("Height: ");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.height)
                                .speed(8.0)
                                .clamp_range(8..=4096),
                        )
                        .on_hover_text("Must be divisible by 8");
                        ui.label("Iterations: ");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.iters)
                                .speed(0.5)
                                .clamp_range(1..=128),
                        )
                        .on_hover_text("Usually 16~64 is enough");
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.label("Seed: ");
                        ui.add(
                            egui::widgets::DragValue::new(&mut self.seed)
                                .speed(u64::MAX.to_f64() / 16384.0)
                                .clamp_range(0..=u64::MAX),
                        );
                        if ui.button("Generate Seed").clicked() {
                            self.seed = rand::random();
                        }
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if ui.button("Generate Image").clicked() {
                            if self.hthread.is_some() {
                                g_bmsg = true;
                            } else {
                                let (sendchannel, recvchannel) = std::sync::mpsc::channel();
                                self.recvchannel = Some(recvchannel);
                                let pprompt = self.positive_prompt.clone();
                                let nprompt = self.negative_prompt.clone();
                                let w = self.width;
                                let h = self.height;
                                let steps = self.iters;
                                let seed = self.seed;
                                let ds = if self.use_dml {
                                    DiffusionDeviceControl {
                                        unet: DiffusionDevice::DirectML(0),
                                        ..Default::default()
                                    }
                                } else {
                                    DiffusionDeviceControl::default()
                                };
                                let low_mem = self.use_low_mem;
                                self.timer = std::time::Instant::now();
                                self.hthread = Some(thread::spawn(move || {
                                    let ortenv = Arc::new(
                                        match OrtEnvironment::builder()
                                            .with_name("Stable Diffusion")
                                            .build()
                                        {
                                            Ok(e) => e,
                                            Err(e) => {
                                                simple_message_box::create_message_box(
                                                    &format!("Cannot init ORT environment: {}", e),
                                                    "Error",
                                                );
                                                panic!("Cannot init ORT environment: {}", e);
                                            }
                                        },
                                    );
                                    let mut scheduler =
                                        match DDIMScheduler::stable_diffusion_v1_optimized_default()
                                        {
                                            Ok(s) => s,
                                            Err(e) => {
                                                simple_message_box::create_message_box(
                                                    &format!("Cannot init scheduler: {}", e),
                                                    "Error",
                                                );
                                                panic!("Cannot init scheduler: {}", e);
                                            }
                                        };
                                    let imgs;
                                    if low_mem {
                                        let pipeline =
                                            match StableDiffusionMemoryOptimizedPipeline::new(
                                                &ortenv,
                                                std::env::current_exe()
                                                    .expect("Cannot get current exe path.")
                                                    .parent()
                                                    .expect("Cannot get parent path.")
                                                    .join("models"),
                                                StableDiffusionOptions {
                                                    devices: ds,
                                                    ..Default::default()
                                                },
                                            ) {
                                                Ok(p) => p,
                                                Err(e) => {
                                                    simple_message_box::create_message_box(
                                                        &format!("Cannot init pipeline: {}", e),
                                                        "Error",
                                                    );
                                                    panic!("Cannot init pipeline: {}", e);
                                                }
                                            };
                                        imgs = pipeline.txt2img(
                                            pprompt,
                                            &mut scheduler,
                                            StableDiffusionTxt2ImgOptions {
                                                width: w,
                                                height: h,
                                                steps: steps,
                                                seed: Some(seed),
                                                negative_prompt: Some(Prompt::from(nprompt)),
                                                ..Default::default()
                                            },
                                        );
                                    } else {
                                        let pipeline = match StableDiffusionPipeline::new(
                                            &ortenv,
                                            std::env::current_exe()
                                                .expect("Cannot get current exe path.")
                                                .parent()
                                                .expect("Cannot get parent path.")
                                                .join("models"),
                                            StableDiffusionOptions {
                                                devices: ds,
                                                ..Default::default()
                                            },
                                        ) {
                                            Ok(p) => p,
                                            Err(e) => {
                                                simple_message_box::create_message_box(
                                                    &format!("Cannot init pipeline: {}", e),
                                                    "Error",
                                                );
                                                panic!("Cannot init pipeline: {}", e);
                                            }
                                        };
                                        imgs = pipeline.txt2img(
                                            pprompt,
                                            &mut scheduler,
                                            StableDiffusionTxt2ImgOptions {
                                                width: w,
                                                height: h,
                                                steps: steps,
                                                seed: Some(seed),
                                                negative_prompt: Some(Prompt::from(nprompt)),
                                                ..Default::default()
                                            },
                                        );
                                    };

                                    let img = match imgs {
                                        Ok(imgs) => imgs[0].to_rgb8(),
                                        Err(e) => {
                                            simple_message_box::create_message_box(
                                                &format!("Cannot generate image: {}", e),
                                                "Error",
                                            );
                                            panic!("Cannot generate image: {}", e);
                                        }
                                    };
                                    let mut bytes: Vec<u8> = Vec::new();
                                    if let Err(e) = img.write_to(
                                        &mut Cursor::new(&mut bytes),
                                        image::ImageOutputFormat::Png,
                                    ) {
                                        simple_message_box::create_message_box(
                                            &format!("Cannot write image: {}", e),
                                            "Error",
                                        );
                                        panic!("Cannot write image: {}", e);
                                    };
                                    if let Err(e) = sendchannel.send(bytes) {
                                        simple_message_box::create_message_box(
                                            &format!("Cannot send image: {}", e),
                                            "Error",
                                        );
                                        panic!("Cannot send image: {}", e);
                                    };
                                }));
                            }
                        }
                        if ui.button("Save Image").clicked() {
                            if self.image.is_none() {
                                s_bmsgnf = true;
                            } else {
                                s_bmsg = true;
                            }
                        }
                        if self.hthread.is_some() {
                            ui.spinner();
                            ui.label("Generating...");
                        }
                    });
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        ui.label("Device: ");
                        if ui
                            .button(if self.use_dml { "DirectML" } else { "CPU" })
                            .clicked()
                        {
                            self.use_dml = !self.use_dml;
                        };
                        ui.label("Low Memory: ");
                        if ui
                            .button(if self.use_low_mem { "ON " } else { "OFF" })
                            .on_hover_text(if self.use_low_mem {
                                "Use low memory mode, but slower."
                            } else {
                                "Use high memory mode, but faster."
                            })
                            .clicked()
                        {
                            self.use_low_mem = !self.use_low_mem;
                        };
                    });
                    ui.add_space(200.0);
                    ui.label("Tips: use () to increase the weight and [] to decrease.");
                });
                ui.separator();
                ui.vertical(|ui| {
                    ui.heading("Preview:");
                    if let Some(img) = &self.image {
                        let w = img.width() as f32;
                        let h = img.height() as f32;
                        let w_available = ui.available_width();
                        let h_available = ui.available_height();
                        let scale = (w_available / w).min(h_available / h).min(1.0);
                        img.show_scaled(ui, scale);
                    } else {
                        let w = self.no_img.width() as f32;
                        let h = self.no_img.height() as f32;
                        let w_available = ui.available_width();
                        let h_available = ui.available_height();
                        let scale = (w_available / w).min(h_available / h).min(1.0);
                        self.no_img.show_scaled(ui, scale);
                    }
                });
            });
        });

        if g_bmsg {
            simple_message_box::create_message_box("Already running.", "Warning");
        }
        if s_bmsg {
            let file = rfd::FileDialog::new()
                .add_filter("PNG File", &["png"])
                .save_file();
            if let Some(f) = file {
                // save raw png to file
                if let Some(data) = &self.rawpng {
                    if let Ok(mut fs) = std::fs::File::create(f) {
                        s_err = fs.write(data).err();
                    } else {
                        s_err = Some(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            "Cannot creat file.",
                        ));
                    }
                }
            }
        }
        if s_bmsgnf {
            simple_message_box::create_message_box("No Image to Save.", "Warning");
        }
        if p_bmsg {
            let file = rfd::FileDialog::new()
                .add_filter("TXT", &["txt"])
                .pick_file();
            if let Some(f) = file {
                match std::fs::read_to_string(f) {
                    Ok(s) => self.positive_prompt = s,
                    Err(e) => simple_message_box::create_message_box(
                        &format!("Error when reading file: {}", e),
                        "Error",
                    ),
                }
            }
        }
        if n_bmsg {
            let file = rfd::FileDialog::new()
                .add_filter("TXT", &["txt"])
                .pick_file();
            if let Some(f) = file {
                match std::fs::read_to_string(f) {
                    Ok(s) => self.negative_prompt = s,
                    Err(e) => simple_message_box::create_message_box(
                        &format!("Error when reading file: {}", e),
                        "Error",
                    ),
                }
            }
        }
        if let Some(e) = s_err {
            simple_message_box::create_message_box(
                &format!("Error when writing file: {}", e),
                "Error",
            );
        }
        if let Some(recv) = &self.recvchannel {
            if let Ok(data) = recv.try_recv() {
                self.image = egui_extras::RetainedImage::from_image_bytes("Image", &data).ok();
                self.rawpng = Some(data);
                self.hthread = None;
                self.recvchannel = None;
                simple_message_box::create_message_box(
                    &format!(
                        "Image Generated in:\r\n {} s, {} s/iter",
                        self.timer.elapsed().as_secs_f32(),
                        self.timer.elapsed().as_secs_f32() / (self.iters as f32)
                    ),
                    "Done",
                );
            }
        }
    }
}
