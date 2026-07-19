// serenity-eguitrainer — native egui frontend for the serenity web trainer.
//
// A SECOND frontend on the SAME supervisor API the browser UI uses
// (webui/src/main.rs; spec docs/UI_MAP_2026-07-05.md §9). No launch logic
// lives here: presets, config merging, argv shapes, spawn, progress parsing,
// and the board DB are all server-side. This app is forms + charts + galleries
// over /api/*, so it works against a remote trainer box too (Settings → base URL).
//
// Threading: ureq is blocking, so ALL HTTP happens on worker threads
// (a 2s poller, an SSE tail, an image loader, and one-shot action threads).
// The UI thread only drains an mpsc channel.

mod api;

use eframe::egui;
use egui::{Color32, ColorImage, RichText, TextureHandle};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::io::{BufRead, BufReader};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

const DEFAULT_BASE: &str = "http://127.0.0.1:8188";
const LOG_CAP: usize = 2000;
const TEXTURE_CAP: usize = 128;

fn settings_path() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.config/serenity_eguitrainer.json")
}

// ---------------------------------------------------------------- messages

enum Msg {
    Presets(Value),
    Runs(Value),
    Metrics(Value),
    Event(Value),
    History(Value),
    Samples(u64, Value),
    Image(String, ColorImage),
    DatasetItems(Value),
    CaptionText(String, String),
    CaptionerStatus(Value),
    BoardRuns(Value),
    BoardTags(String, Vec<String>),
    BoardSeries(String, String, Vec<[f64; 2]>),
    BoardHparams(String, Value),
    ActionResult(String),
    DryRun(Value),
    Validations(String, String),
    Error(String),
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    Train,
    Dataset,
    Captioner,
    Validations,
    Samples,
    Board,
    Runs,
    Logs,
    Settings,
}

const SECTIONS: &[(Section, &str)] = &[
    (Section::Train, "Train"),
    (Section::Dataset, "Dataset"),
    (Section::Captioner, "Captioner"),
    (Section::Validations, "Validations"),
    (Section::Samples, "Samples"),
    (Section::Board, "Board"),
    (Section::Runs, "Runs"),
    (Section::Logs, "Logs"),
    (Section::Settings, "Settings"),
];

// ---------------------------------------------------------------- app state

struct App {
    base: Arc<Mutex<String>>,
    base_edit: String,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    img_tx: Sender<String>,
    requested_imgs: HashSet<String>,
    textures: HashMap<String, TextureHandle>,
    texture_order: VecDeque<String>,

    section: Section,
    status_line: String,

    // Train tab
    presets: Vec<Value>,
    sel_preset: usize,
    overrides: BTreeMap<String, Value>,
    run_name: String,
    cache: String,
    resume_state: String,
    start_step: String,
    dry: Option<Value>,

    // live state
    runs: Vec<Value>,
    gpu_busy: Option<String>,
    metrics: Value,
    live: Option<Value>,
    logs: VecDeque<String>,
    loss_series: Vec<[f64; 2]>,
    grad_series: Vec<[f64; 2]>,
    speed_series: Vec<[f64; 2]>,

    // Samples tab
    samples: Vec<String>,
    samples_run: Option<u64>,
    lightbox: Option<String>,

    // Dataset tab
    dataset_path: String,
    dataset_items: Vec<Value>,
    sel_caption: Option<String>,
    caption_text: String,

    // Captioner tab
    cap_folder: String,
    cap_model: String,
    cap_prompt: String,
    cap_max_tokens: u64,
    cap_skip: bool,
    cap_one: bool,
    cap_status: Value,

    // Validations tab
    val_path: String,
    val_text: String,

    // Board tab
    board_runs: Vec<Value>,
    board_sel: Option<String>,
    board_series: BTreeMap<String, Vec<[f64; 2]>>,
    board_hparams: Value,

    // Runs tab
    history: Vec<Value>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Persisted {
    base_url: Option<String>,
    dataset_path: Option<String>,
    cap_folder: Option<String>,
    val_path: Option<String>,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let persisted: Persisted = std::fs::read_to_string(settings_path())
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let base_url = persisted.base_url.unwrap_or_else(|| DEFAULT_BASE.into());
        let base = Arc::new(Mutex::new(base_url.clone()));
        let (tx, rx) = channel::<Msg>();
        let (img_tx, img_rx) = channel::<String>();
        let ctx = cc.egui_ctx.clone();

        spawn_poller(base.clone(), tx.clone(), ctx.clone());
        spawn_sse(base.clone(), tx.clone(), ctx.clone());
        spawn_image_loader(base.clone(), img_rx, tx.clone(), ctx.clone());

        Self {
            base,
            base_edit: base_url,
            tx,
            rx,
            img_tx,
            requested_imgs: HashSet::new(),
            textures: HashMap::new(),
            texture_order: VecDeque::new(),
            section: Section::Train,
            status_line: "connecting…".into(),
            presets: vec![],
            sel_preset: 0,
            overrides: BTreeMap::new(),
            run_name: String::new(),
            cache: String::new(),
            resume_state: String::new(),
            start_step: String::new(),
            dry: None,
            runs: vec![],
            gpu_busy: None,
            metrics: json!({}),
            live: None,
            logs: VecDeque::new(),
            loss_series: vec![],
            grad_series: vec![],
            speed_series: vec![],
            samples: vec![],
            samples_run: None,
            lightbox: None,
            dataset_path: persisted.dataset_path.unwrap_or_default(),
            dataset_items: vec![],
            sel_caption: None,
            caption_text: String::new(),
            cap_folder: persisted.cap_folder.unwrap_or_default(),
            cap_model: "Qwen/Qwen3-VL-8B-Instruct".into(),
            cap_prompt: "Describe this media.".into(),
            cap_max_tokens: 128,
            cap_skip: true,
            cap_one: false,
            cap_status: json!({}),
            val_path: persisted.val_path.unwrap_or_default(),
            val_text: String::new(),
            board_runs: vec![],
            board_sel: None,
            board_series: BTreeMap::new(),
            board_hparams: json!({}),
            history: vec![],
        }
    }

    fn save_settings(&self) {
        let p = Persisted {
            base_url: Some(self.base_edit.clone()),
            dataset_path: Some(self.dataset_path.clone()),
            cap_folder: Some(self.cap_folder.clone()),
            val_path: Some(self.val_path.clone()),
        };
        if let Some(dir) = std::path::Path::new(&settings_path()).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(settings_path(), serde_json::to_string_pretty(&p).unwrap_or_default());
    }

    /// Fire a one-shot request on a worker thread; result comes back as Msg.
    fn action<F>(&self, f: F)
    where
        F: FnOnce(&str) -> Msg + Send + 'static,
    {
        let base = self.base.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let b = base.lock().unwrap().clone();
            let _ = tx.send(f(&b));
        });
    }

    fn request_image(&mut self, url_path: &str) {
        if self.textures.contains_key(url_path) || self.requested_imgs.contains(url_path) {
            return;
        }
        self.requested_imgs.insert(url_path.to_string());
        let _ = self.img_tx.send(url_path.to_string());
    }

    fn active_run_id(&self) -> Option<u64> {
        self.runs
            .iter()
            .find(|r| r["status"] == "running")
            .and_then(|r| r["id"].as_u64())
    }

    // ------------------------------------------------------------ messages

    fn drain(&mut self, ctx: &egui::Context) {
        while let Ok(m) = self.rx.try_recv() {
            match m {
                Msg::Presets(v) => {
                    self.presets = v["presets"].as_array().cloned().unwrap_or_default();
                    if !self.presets.is_empty() && self.overrides.is_empty() {
                        self.load_preset(0);
                    }
                    self.status_line = format!("connected — {} presets", self.presets.len());
                }
                Msg::Runs(v) => {
                    self.runs = v["runs"].as_array().cloned().unwrap_or_default();
                    self.gpu_busy = v["gpu_busy"].as_str().map(|s| s.to_string());
                }
                Msg::Metrics(v) => self.metrics = v,
                Msg::History(v) => self.history = v["history"].as_array().cloned().unwrap_or_default(),
                Msg::Samples(id, v) => {
                    self.samples_run = Some(id);
                    self.samples = v["samples"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|s| s.as_str().map(String::from)).collect())
                        .unwrap_or_default();
                }
                Msg::Image(url, img) => {
                    let tex = ctx.load_texture(&url, img, Default::default());
                    self.textures.insert(url.clone(), tex);
                    self.texture_order.push_back(url);
                    while self.texture_order.len() > TEXTURE_CAP {
                        if let Some(old) = self.texture_order.pop_front() {
                            self.textures.remove(&old);
                            self.requested_imgs.remove(&old);
                        }
                    }
                }
                Msg::DatasetItems(v) => {
                    if let Some(e) = v["error"].as_str() {
                        self.status_line = e.to_string();
                        self.dataset_items = vec![];
                    } else {
                        self.dataset_items = v["items"].as_array().cloned().unwrap_or_default();
                        self.status_line = format!("{} images", self.dataset_items.len());
                    }
                }
                Msg::CaptionText(path, text) => {
                    self.sel_caption = Some(path);
                    self.caption_text = text;
                }
                Msg::CaptionerStatus(v) => self.cap_status = v,
                Msg::BoardRuns(v) => self.board_runs = v.as_array().cloned().unwrap_or_default(),
                Msg::BoardTags(run, tags) => {
                    if Some(&run) == self.board_sel.as_ref() {
                        for tag in tags.into_iter().take(6) {
                            let r = run.clone();
                            self.action(move |b| {
                                match api::get_json(
                                    b,
                                    &format!("/api/board/runs/{}/scalars?tag={}&downsample=2000", urlenc(&r), urlenc(&tag)),
                                ) {
                                    Ok(v) => Msg::BoardSeries(r, tag, parse_series(&v)),
                                    Err(e) => Msg::Error(e.to_string()),
                                }
                            });
                        }
                    }
                }
                Msg::BoardSeries(run, tag, pts) => {
                    if Some(&run) == self.board_sel.as_ref() {
                        self.board_series.insert(tag, pts);
                    }
                }
                Msg::BoardHparams(run, v) => {
                    if Some(&run) == self.board_sel.as_ref() {
                        self.board_hparams = v;
                    }
                }
                Msg::ActionResult(s) => self.status_line = s,
                Msg::DryRun(v) => self.dry = Some(v),
                Msg::Validations(path, text) => {
                    self.val_path = path;
                    self.val_text = text;
                }
                Msg::Event(v) => self.handle_event(v),
                Msg::Error(e) => self.status_line = e,
            }
        }
    }

    fn handle_event(&mut self, v: Value) {
        match v["type"].as_str().unwrap_or("") {
            "progress" => {
                let run = v["run"].clone();
                let step = run["step"].as_f64().unwrap_or(0.0);
                let last = self.loss_series.last().map(|p| p[0]).unwrap_or(-1.0);
                if step > last {
                    self.loss_series.push([step, run["loss"].as_f64().unwrap_or(0.0)]);
                    self.grad_series.push([step, run["grad_norm"].as_f64().unwrap_or(0.0)]);
                    self.speed_series.push([step, run["s_per_step"].as_f64().unwrap_or(0.0)]);
                }
                self.live = Some(run);
            }
            "status" => {
                let run = v["run"].clone();
                self.status_line = format!(
                    "run #{} {}",
                    run["id"].as_u64().unwrap_or(0),
                    run["status"].as_str().unwrap_or("?")
                );
                self.live = Some(run);
            }
            "log" => {
                if let Some(line) = v["line"].as_str() {
                    self.logs.push_back(line.to_string());
                    while self.logs.len() > LOG_CAP {
                        self.logs.pop_front();
                    }
                }
            }
            _ => {}
        }
    }

    fn load_preset(&mut self, idx: usize) {
        self.sel_preset = idx;
        let Some(p) = self.presets.get(idx) else { return };
        self.overrides.clear();
        if let Some(r) = p["recipe"].as_object() {
            for (k, v) in r {
                self.overrides.insert(k.clone(), v.clone());
            }
        }
        self.run_name = p["run_name"].as_str().unwrap_or("").to_string();
        self.cache = p["cache"].as_str().unwrap_or("").to_string();
        self.dry = None;
    }

    fn launch_body(&self, dry: bool) -> Value {
        let mut body = json!({
            "preset_id": self.presets.get(self.sel_preset).and_then(|p| p["id"].as_str()).unwrap_or(""),
            "overrides": Value::Object(self.overrides.clone().into_iter().collect()),
            "dry_run": dry,
        });
        if !self.run_name.trim().is_empty() {
            body["run_name"] = json!(self.run_name.trim());
        }
        if !self.cache.trim().is_empty() {
            body["cache"] = json!(self.cache.trim());
        }
        if !self.resume_state.trim().is_empty() {
            body["resume_state"] = json!(self.resume_state.trim());
            if let Ok(s) = self.start_step.trim().parse::<u64>() {
                body["start_step"] = json!(s);
            }
        }
        body
    }
}

// ---------------------------------------------------------------- workers

fn spawn_poller(base: Arc<Mutex<String>>, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::spawn(move || {
        let mut tick: u64 = 0;
        // fetch presets until they land
        loop {
            let b = base.lock().unwrap().clone();
            let mut dirty = false;
            if tick % 5 == 0 || tick == 0 {
                match api::get_json(&b, "/api/presets") {
                    Ok(v) => {
                        if tick == 0 || tick % 30 == 0 {
                            let _ = tx.send(Msg::Presets(v));
                            dirty = true;
                        }
                    }
                    Err(e) => {
                        if tick % 10 == 0 {
                            let _ = tx.send(Msg::Error(format!("server unreachable: {e}")));
                            dirty = true;
                        }
                    }
                }
            }
            if let Ok(v) = api::get_json(&b, "/api/runs") {
                // remember the running id for the samples poll below
                let running = v["runs"]
                    .as_array()
                    .and_then(|a| a.iter().find(|r| r["status"] == "running").and_then(|r| r["id"].as_u64()));
                let _ = tx.send(Msg::Runs(v));
                dirty = true;
                if let Some(id) = running {
                    if tick % 3 == 0 {
                        if let Ok(s) = api::get_json(&b, &format!("/api/runs/{id}/samples")) {
                            let _ = tx.send(Msg::Samples(id, s));
                        }
                    }
                }
            }
            if tick % 2 == 0 {
                if let Ok(v) = api::get_json(&b, "/api/system/metrics") {
                    let _ = tx.send(Msg::Metrics(v));
                    dirty = true;
                }
                if let Ok(v) = api::get_json(&b, "/api/captioner/status") {
                    let _ = tx.send(Msg::CaptionerStatus(v));
                }
            }
            if tick % 10 == 0 {
                if let Ok(v) = api::get_json(&b, "/api/runs/history") {
                    let _ = tx.send(Msg::History(v));
                }
                if let Ok(v) = api::get_json(&b, "/api/board/runs") {
                    let _ = tx.send(Msg::BoardRuns(v));
                }
            }
            if dirty {
                ctx.request_repaint();
            }
            tick += 1;
            std::thread::sleep(Duration::from_millis(2000));
        }
    });
}

fn spawn_sse(base: Arc<Mutex<String>>, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::spawn(move || loop {
        let b = base.lock().unwrap().clone();
        match api::open_sse(&b) {
            Ok(reader) => {
                let br = BufReader::new(reader);
                for line in br.lines() {
                    let Ok(line) = line else { break };
                    if let Some(payload) = line.strip_prefix("data:") {
                        if let Ok(v) = serde_json::from_str::<Value>(payload.trim()) {
                            let _ = tx.send(Msg::Event(v));
                            ctx.request_repaint();
                        }
                    }
                    // base URL changed under us -> reconnect
                    if *base.lock().unwrap() != b {
                        break;
                    }
                }
            }
            Err(_) => {}
        }
        std::thread::sleep(Duration::from_secs(2));
    });
}

fn spawn_image_loader(base: Arc<Mutex<String>>, rx: Receiver<String>, tx: Sender<Msg>, ctx: egui::Context) {
    std::thread::spawn(move || {
        while let Ok(url_path) = rx.recv() {
            let b = base.lock().unwrap().clone();
            match api::get_bytes(&b, &url_path) {
                Ok(bytes) => match image::load_from_memory(&bytes) {
                    Ok(img) => {
                        // keep textures bounded: max dim 1024 (grid shows ~320px, lightbox scales)
                        let img = if img.width().max(img.height()) > 1024 {
                            img.thumbnail(1024, 1024)
                        } else {
                            img
                        };
                        let rgba = img.to_rgba8();
                        let size = [rgba.width() as usize, rgba.height() as usize];
                        let ci = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
                        let _ = tx.send(Msg::Image(url_path, ci));
                        ctx.request_repaint();
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Error(format!("decode {url_path}: {e}")));
                    }
                },
                Err(e) => {
                    let _ = tx.send(Msg::Error(format!("{e}")));
                }
            }
        }
    });
}

// ---------------------------------------------------------------- chart

fn chart(ui: &mut egui::Ui, label: &str, series: &[[f64; 2]], color: Color32, height: f32) {
    let width = ui.available_width();
    let (resp, painter) = ui.allocate_painter(egui::vec2(width, height), egui::Sense::hover());
    let rect = resp.rect;
    painter.rect_filled(rect, 3.0, ui.visuals().extreme_bg_color);
    if series.len() < 2 {
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            format!("{label} — waiting for data"),
            egui::FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
        return;
    }
    let (mut xmin, mut xmax) = (f64::MAX, f64::MIN);
    let (mut ymin, mut ymax) = (f64::MAX, f64::MIN);
    for p in series {
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    if (xmax - xmin).abs() < 1e-12 {
        xmax = xmin + 1.0;
    }
    if (ymax - ymin).abs() < 1e-12 {
        ymax = ymin + 1.0;
    }
    let pad = 4.0;
    let to_pos = |p: &[f64; 2]| {
        let fx = ((p[0] - xmin) / (xmax - xmin)) as f32;
        let fy = ((p[1] - ymin) / (ymax - ymin)) as f32;
        egui::pos2(
            rect.left() + pad + fx * (rect.width() - 2.0 * pad),
            rect.bottom() - pad - fy * (rect.height() - 2.0 * pad),
        )
    };
    let pts: Vec<egui::Pos2> = series.iter().map(to_pos).collect();
    painter.add(egui::Shape::line(pts, egui::Stroke::new(1.5, color)));
    let last = series.last().unwrap();
    painter.text(
        rect.left_top() + egui::vec2(6.0, 4.0),
        egui::Align2::LEFT_TOP,
        format!("{label}  last {:.5}  min {:.5}  max {:.5}", last[1], ymin, ymax),
        egui::FontId::proportional(11.0),
        ui.visuals().text_color(),
    );
    // hover readout: nearest point by x
    if let Some(hp) = resp.hover_pos() {
        let fx = ((hp.x - rect.left() - pad) / (rect.width() - 2.0 * pad)).clamp(0.0, 1.0) as f64;
        let x = xmin + fx * (xmax - xmin);
        let nearest = series
            .iter()
            .min_by(|a, b| (a[0] - x).abs().partial_cmp(&(b[0] - x).abs()).unwrap())
            .unwrap();
        painter.circle_filled(to_pos(nearest), 3.0, color);
        painter.text(
            rect.right_top() + egui::vec2(-6.0, 4.0),
            egui::Align2::RIGHT_TOP,
            format!("step {:.0}: {:.6}", nearest[0], nearest[1]),
            egui::FontId::proportional(11.0),
            ui.visuals().strong_text_color(),
        );
    }
}

// ---------------------------------------------------------------- tabs

impl App {
    fn ui_train(&mut self, ui: &mut egui::Ui) {
        ui.heading("Train");
        ui.add_space(4.0);
        let labels: Vec<String> = self
            .presets
            .iter()
            .map(|p| {
                let wired = p["wired"].as_bool().unwrap_or(false);
                format!(
                    "{}{}",
                    p["label"].as_str().unwrap_or("?"),
                    if wired { "" } else { "  [not wired]" }
                )
            })
            .collect();
        let mut changed_to: Option<usize> = None;
        ui.horizontal(|ui| {
            ui.label("Preset");
            egui::ComboBox::from_id_salt("preset")
                .width(260.0)
                .selected_text(labels.get(self.sel_preset).cloned().unwrap_or_else(|| "—".into()))
                .show_ui(ui, |ui| {
                    for (i, l) in labels.iter().enumerate() {
                        if ui.selectable_label(i == self.sel_preset, l).clicked() {
                            changed_to = Some(i);
                        }
                    }
                });
            if let Some(p) = self.presets.get(self.sel_preset) {
                ui.weak(format!(
                    "backend {} · argv {}",
                    p["backend"].as_str().unwrap_or("?"),
                    p["argv_shape"].as_str().unwrap_or("?")
                ));
            }
        });
        if let Some(i) = changed_to {
            self.load_preset(i);
        }
        ui.add_space(6.0);
        egui::Grid::new("run_fields").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Run name");
            ui.add(egui::TextEdit::singleline(&mut self.run_name).desired_width(420.0));
            ui.end_row();
            ui.label("Cache");
            ui.add(egui::TextEdit::singleline(&mut self.cache).desired_width(420.0));
            ui.end_row();
            ui.label("Resume .state");
            ui.add(egui::TextEdit::singleline(&mut self.resume_state).desired_width(420.0));
            ui.end_row();
            ui.label("Start step");
            ui.add(egui::TextEdit::singleline(&mut self.start_step).desired_width(120.0));
            ui.end_row();
        });
        ui.add_space(6.0);
        ui.separator();
        ui.strong("Recipe (preset values — edits are sent as overrides)");
        ui.add_space(2.0);
        egui::Grid::new("recipe").num_columns(2).spacing([8.0, 4.0]).striped(true).show(ui, |ui| {
            let keys: Vec<String> = self.overrides.keys().cloned().collect();
            for k in keys {
                ui.label(&k);
                let v = self.overrides.get_mut(&k).unwrap();
                match v {
                    Value::Number(n) => {
                        if n.is_u64() || n.is_i64() {
                            let mut x = n.as_i64().unwrap_or(0);
                            if ui.add(egui::DragValue::new(&mut x).speed(1)).changed() {
                                *v = json!(x);
                            }
                        } else {
                            let mut x = n.as_f64().unwrap_or(0.0);
                            if ui
                                .add(egui::DragValue::new(&mut x).speed(0.00001).max_decimals(8))
                                .changed()
                            {
                                *v = json!(x);
                            }
                        }
                    }
                    Value::Bool(b) => {
                        let mut x = *b;
                        if ui.checkbox(&mut x, "").changed() {
                            *v = json!(x);
                        }
                    }
                    Value::String(s) => {
                        let mut x = s.clone();
                        if ui.add(egui::TextEdit::singleline(&mut x).desired_width(280.0)).changed() {
                            *v = json!(x);
                        }
                    }
                    other => {
                        ui.weak(other.to_string());
                    }
                }
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            let wired = self
                .presets
                .get(self.sel_preset)
                .and_then(|p| p["wired"].as_bool())
                .unwrap_or(false);
            let running = self.active_run_id();
            let can_start = wired && running.is_none();
            if ui
                .add_enabled(can_start, egui::Button::new(RichText::new("▶ Start").strong()))
                .clicked()
            {
                let body = self.launch_body(false);
                self.action(move |b| match api::post_json(b, "/api/runs", &body) {
                    Ok((200, v)) => Msg::ActionResult(format!(
                        "started run #{} — {}",
                        v["run_id"].as_u64().unwrap_or(0),
                        v["workspace"].as_str().unwrap_or("")
                    )),
                    Ok((code, v)) => Msg::ActionResult(format!("launch {code}: {v}")),
                    Err(e) => Msg::Error(e.to_string()),
                });
            }
            if ui.button("Dry run (show argv + config)").clicked() {
                let body = self.launch_body(true);
                self.action(move |b| match api::post_json(b, "/api/runs", &body) {
                    Ok((200, v)) => Msg::DryRun(v),
                    Ok((code, v)) => Msg::ActionResult(format!("dry-run {code}: {v}")),
                    Err(e) => Msg::Error(e.to_string()),
                });
            }
            if let Some(id) = running {
                if ui.button(RichText::new("■ Stop").color(Color32::from_rgb(220, 80, 80))).clicked() {
                    self.action(move |b| match api::post_json(b, &format!("/api/runs/{id}/stop"), &json!({})) {
                        Ok((_, v)) => Msg::ActionResult(format!("stop: {v}")),
                        Err(e) => Msg::Error(e.to_string()),
                    });
                }
            }
            if let Some(busy) = &self.gpu_busy {
                ui.colored_label(Color32::from_rgb(230, 160, 60), format!("GPU busy: {busy}"));
            }
        });
        if let Some(d) = &self.dry {
            ui.add_space(6.0);
            egui::CollapsingHeader::new("Dry-run result").default_open(true).show(ui, |ui| {
                ui.monospace(format!(
                    "binary: {}\nargs: {}",
                    d["binary"].as_str().unwrap_or("?"),
                    d["args"]
                        .as_array()
                        .map(|a| a.iter().filter_map(|x| x.as_str()).collect::<Vec<_>>().join(" "))
                        .unwrap_or_default()
                ));
                ui.weak(format!("config written to {}", d["config_written"].as_str().unwrap_or("?")));
                egui::ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                    ui.monospace(serde_json::to_string_pretty(&d["config"]).unwrap_or_default());
                });
            });
        }
    }

    fn ui_dataset(&mut self, ui: &mut egui::Ui) {
        ui.heading("Dataset");
        ui.horizontal(|ui| {
            ui.label("Folder");
            ui.add(egui::TextEdit::singleline(&mut self.dataset_path).desired_width(480.0));
            if ui.button("Scan").clicked() {
                self.save_settings();
                let path = self.dataset_path.clone();
                self.action(move |b| {
                    match api::get_json(b, &format!("/api/dataset/media?path={}", urlenc(&path))) {
                        Ok(v) => Msg::DatasetItems(v),
                        Err(e) => Msg::Error(e.to_string()),
                    }
                });
            }
        });
        ui.add_space(4.0);
        // caption editor for the selected image
        if let Some(cap_path) = self.sel_caption.clone() {
            ui.group(|ui| {
                ui.strong(format!("Caption — {cap_path}"));
                ui.add(
                    egui::TextEdit::multiline(&mut self.caption_text)
                        .desired_rows(3)
                        .desired_width(f32::INFINITY),
                );
                ui.horizontal(|ui| {
                    if ui.button("Save caption").clicked() {
                        let body = json!({"path": cap_path, "text": self.caption_text});
                        self.action(move |b| match api::put_json(b, "/api/caption", &body) {
                            Ok((200, _)) => Msg::ActionResult("caption saved".into()),
                            Ok((code, v)) => Msg::ActionResult(format!("caption {code}: {v}")),
                            Err(e) => Msg::Error(e.to_string()),
                        });
                    }
                    if ui.button("Close").clicked() {
                        self.sel_caption = None;
                    }
                });
            });
            ui.add_space(4.0);
        }
        let items = self.dataset_items.clone();
        self.gallery(ui, &items);
    }

    /// Shared thumbnail grid. `items` rows: {"image": url, "path": abs, "caption_path": opt}
    fn gallery(&mut self, ui: &mut egui::Ui, items: &[Value]) {
        let cols = ((ui.available_width() / 340.0).floor() as usize).clamp(2, 8);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            egui::Grid::new("gallery").num_columns(cols).spacing([8.0, 8.0]).show(ui, |ui| {
                for (i, it) in items.iter().enumerate() {
                    let url = it["image"].as_str().unwrap_or("").to_string();
                    self.request_image(&url);
                    ui.vertical(|ui| {
                        let size = egui::vec2(320.0, 240.0);
                        if let Some(tex) = self.textures.get(&url) {
                            let img = egui::Image::new(tex).fit_to_exact_size(size).sense(egui::Sense::click());
                            let resp = ui.add(img);
                            if resp.clicked() {
                                self.lightbox = Some(url.clone());
                            }
                        } else {
                            let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                            ui.painter().rect_filled(rect, 3.0, ui.visuals().faint_bg_color);
                            ui.painter().text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "loading…",
                                egui::FontId::proportional(12.0),
                                ui.visuals().weak_text_color(),
                            );
                        }
                        let name = it["path"]
                            .as_str()
                            .and_then(|p| p.rsplit('/').next())
                            .unwrap_or_else(|| url.rsplit('/').next().unwrap_or("?"));
                        ui.horizontal(|ui| {
                            ui.weak(name);
                            if let Some(cp) = it["caption_path"].as_str() {
                                let cp = cp.to_string();
                                if ui.small_button("caption").clicked() {
                                    self.action(move |b| {
                                        match api::get_json(b, &format!("/api/caption?path={}", urlenc(&cp))) {
                                            Ok(v) => Msg::CaptionText(
                                                v["path"].as_str().unwrap_or("").into(),
                                                v["text"].as_str().unwrap_or("").into(),
                                            ),
                                            Err(e) => Msg::Error(e.to_string()),
                                        }
                                    });
                                }
                            }
                        });
                    });
                    if (i + 1) % cols == 0 {
                        ui.end_row();
                    }
                }
            });
        });
    }

    fn ui_captioner(&mut self, ui: &mut egui::Ui) {
        ui.heading("Captioner");
        ui.weak("Qwen-VL folder captioner (server-side, ai-toolkit venv) — writes .txt sidecars.");
        ui.add_space(4.0);
        egui::Grid::new("cap").num_columns(2).spacing([8.0, 6.0]).show(ui, |ui| {
            ui.label("Folder");
            ui.add(egui::TextEdit::singleline(&mut self.cap_folder).desired_width(440.0));
            ui.end_row();
            ui.label("Model");
            ui.add(egui::TextEdit::singleline(&mut self.cap_model).desired_width(440.0));
            ui.end_row();
            ui.label("Prompt");
            ui.add(egui::TextEdit::singleline(&mut self.cap_prompt).desired_width(440.0));
            ui.end_row();
            ui.label("Max tokens");
            ui.add(egui::DragValue::new(&mut self.cap_max_tokens).range(16..=1024));
            ui.end_row();
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.cap_skip, "Skip existing");
            ui.checkbox(&mut self.cap_one, "One sentence");
        });
        ui.add_space(6.0);
        let running = self.cap_status["running"].as_bool().unwrap_or(false);
        ui.horizontal(|ui| {
            if ui.add_enabled(!running, egui::Button::new("▶ Run captioner")).clicked() {
                self.save_settings();
                let body = json!({
                    "folder": self.cap_folder,
                    "model": self.cap_model,
                    "prompt": self.cap_prompt,
                    "max_tokens": self.cap_max_tokens,
                    "skip_existing": self.cap_skip,
                    "one_sentence": self.cap_one,
                });
                self.action(move |b| match api::post_json(b, "/api/captioner/run", &body) {
                    Ok((200, _)) => Msg::ActionResult("captioner started".into()),
                    Ok((code, v)) => Msg::ActionResult(format!("captioner {code}: {v}")),
                    Err(e) => Msg::Error(e.to_string()),
                });
            }
            if running {
                if ui.button("■ Abort").clicked() {
                    self.action(|b| match api::post_json(b, "/api/captioner/abort", &json!({})) {
                        Ok((_, v)) => Msg::ActionResult(format!("abort: {v}")),
                        Err(e) => Msg::Error(e.to_string()),
                    });
                }
            }
        });
        ui.add_space(6.0);
        let s = &self.cap_status;
        if s["running"].as_bool().unwrap_or(false) || s["finished"].as_bool().unwrap_or(false) {
            let done = s["done"].as_u64().unwrap_or(0);
            let total = s["total"].as_u64().unwrap_or(0).max(1);
            ui.add(
                egui::ProgressBar::new(done as f32 / total as f32)
                    .text(format!("{done}/{total}  {}", s["current_file"].as_str().unwrap_or(""))),
            );
            if let Some(err) = s["fatal"].as_str().filter(|e| !e.is_empty()) {
                ui.colored_label(Color32::from_rgb(220, 80, 80), err);
            }
            ui.add_space(4.0);
            egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                if let Some(rs) = s["results"].as_array() {
                    for r in rs.iter().rev().take(50) {
                        ui.horizontal_wrapped(|ui| {
                            ui.strong(r["file"].as_str().and_then(|f| f.rsplit('/').next()).unwrap_or("?"));
                            ui.weak(r["caption"].as_str().unwrap_or(""));
                        });
                    }
                }
            });
        }
    }

    fn ui_validations(&mut self, ui: &mut egui::Ui) {
        ui.heading("Validations (sample-prompts JSON)");
        ui.horizontal(|ui| {
            ui.label("File");
            ui.add(egui::TextEdit::singleline(&mut self.val_path).desired_width(480.0));
            if ui.button("Load").clicked() {
                self.save_settings();
                let path = self.val_path.clone();
                self.action(move |b| {
                    match api::get_json(b, &format!("/api/validations?path={}", urlenc(&path))) {
                        Ok(v) => {
                            if let Some(e) = v["error"].as_str() {
                                Msg::Error(e.to_string())
                            } else {
                                Msg::Validations(
                                    v["path"].as_str().unwrap_or("").into(),
                                    serde_json::to_string_pretty(&v["content"]).unwrap_or_default(),
                                )
                            }
                        }
                        Err(e) => Msg::Error(e.to_string()),
                    }
                });
            }
            if ui.button("Save").clicked() {
                match serde_json::from_str::<Value>(&self.val_text) {
                    Ok(content) => {
                        let body = json!({"path": self.val_path, "content": content});
                        self.action(move |b| match api::put_json(b, "/api/validations", &body) {
                            Ok((200, _)) => Msg::ActionResult("validations saved".into()),
                            Ok((code, v)) => Msg::ActionResult(format!("save {code}: {v}")),
                            Err(e) => Msg::Error(e.to_string()),
                        });
                    }
                    Err(e) => self.status_line = format!("JSON parse error: {e}"),
                }
            }
        });
        ui.weak("server enforces the 1024×1024 minimum for image validation renders");
        ui.add_space(4.0);
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            ui.add(
                egui::TextEdit::multiline(&mut self.val_text)
                    .code_editor()
                    .desired_width(f32::INFINITY)
                    .desired_rows(30),
            );
        });
    }

    fn ui_samples(&mut self, ui: &mut egui::Ui) {
        ui.heading("Samples");
        ui.horizontal(|ui| {
            if let Some(id) = self.samples_run {
                ui.weak(format!("run #{id} — {} images", self.samples.len()));
            } else {
                ui.weak("no run samples yet — appears when a web-launched run writes samples");
            }
            if ui.button("Refresh").clicked() {
                if let Some(id) = self.samples_run.or_else(|| self.runs.last().and_then(|r| r["id"].as_u64())) {
                    self.action(move |b| match api::get_json(b, &format!("/api/runs/{id}/samples")) {
                        Ok(v) => Msg::Samples(id, v),
                        Err(e) => Msg::Error(e.to_string()),
                    });
                }
            }
        });
        ui.add_space(4.0);
        let items: Vec<Value> = self
            .samples
            .iter()
            .map(|u| json!({"image": u, "path": u}))
            .collect();
        self.gallery(ui, &items);
    }

    fn ui_board(&mut self, ui: &mut egui::Ui) {
        ui.heading("Board");
        ui.weak("metrics DB (web + CLI runs) — same data as /board in the browser");
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            let sel_text = self.board_sel.clone().unwrap_or_else(|| "select run…".into());
            let mut pick: Option<String> = None;
            egui::ComboBox::from_id_salt("board_run")
                .width(360.0)
                .selected_text(sel_text)
                .show_ui(ui, |ui| {
                    for r in &self.board_runs {
                        let name = r["name"].as_str().unwrap_or("?").to_string();
                        let label = format!(
                            "{name}  [{} · step {}]",
                            r["status"].as_str().unwrap_or("?"),
                            r["last_step"].as_i64().unwrap_or(0)
                        );
                        if ui.selectable_label(self.board_sel.as_deref() == Some(&name), label).clicked() {
                            pick = Some(name);
                        }
                    }
                });
            if let Some(name) = pick {
                self.board_sel = Some(name.clone());
                self.board_series.clear();
                self.board_hparams = json!({});
                // the DB's tag names are per-source (e.g. loss/train_step) — ask, don't guess
                let run = name.clone();
                self.action(move |b| {
                    match api::get_json(b, &format!("/api/board/runs/{}/tags", urlenc(&run))) {
                        Ok(v) => {
                            let tags = v["scalars"]
                                .as_array()
                                .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                                .unwrap_or_default();
                            Msg::BoardTags(run, tags)
                        }
                        Err(e) => Msg::Error(e.to_string()),
                    }
                });
                let run = name.clone();
                self.action(move |b| {
                    match api::get_json(b, &format!("/api/board/runs/{}/hparams", urlenc(&run))) {
                        Ok(v) => Msg::BoardHparams(run, v),
                        Err(e) => Msg::Error(e.to_string()),
                    }
                });
            }
        });
        ui.add_space(6.0);
        if self.board_sel.is_some() {
            let colors = [
                Color32::from_rgb(90, 170, 255),
                Color32::from_rgb(255, 150, 90),
                Color32::from_rgb(140, 220, 140),
            ];
            let series: Vec<(String, Vec<[f64; 2]>)> =
                self.board_series.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            for (i, (tag, pts)) in series.iter().enumerate() {
                chart(ui, tag, pts, colors[i % colors.len()], 140.0);
                ui.add_space(6.0);
            }
            if let Some(h) = self.board_hparams.as_object().filter(|m| !m.is_empty()) {
                ui.separator();
                ui.strong("hparams");
                egui::Grid::new("hparams").num_columns(2).striped(true).show(ui, |ui| {
                    for (k, v) in h {
                        ui.label(k);
                        ui.monospace(v.to_string());
                        ui.end_row();
                    }
                });
            }
        }
    }

    fn ui_runs(&mut self, ui: &mut egui::Ui) {
        ui.heading("Runs");
        ui.strong("Active");
        egui::Grid::new("active_runs").num_columns(6).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
            for h in ["id", "preset", "status", "step", "loss", "workspace"] {
                ui.strong(h);
            }
            ui.end_row();
            for r in &self.runs {
                ui.label(r["id"].as_u64().unwrap_or(0).to_string());
                ui.label(r["preset_id"].as_str().unwrap_or("?"));
                ui.label(r["status"].as_str().unwrap_or("?"));
                ui.label(format!("{}/{}", r["step"].as_u64().unwrap_or(0), r["total_steps"].as_u64().unwrap_or(0)));
                ui.label(format!("{:.5}", r["loss"].as_f64().unwrap_or(0.0)));
                ui.weak(r["workspace_dir"].as_str().unwrap_or(""));
                ui.end_row();
            }
        });
        ui.add_space(8.0);
        ui.strong("History (last 50)");
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            egui::Grid::new("history").num_columns(6).striped(true).spacing([12.0, 4.0]).show(ui, |ui| {
                for h in ["id", "preset", "status", "step", "loss", "workspace"] {
                    ui.strong(h);
                }
                ui.end_row();
                for r in &self.history {
                    ui.label(r["id"].as_u64().unwrap_or(0).to_string());
                    ui.label(r["preset_id"].as_str().unwrap_or("?"));
                    ui.label(r["status"].as_str().unwrap_or("?"));
                    ui.label(format!(
                        "{}/{}",
                        r["step"].as_u64().unwrap_or(0),
                        r["total_steps"].as_u64().unwrap_or(0)
                    ));
                    ui.label(format!("{:.5}", r["loss"].as_f64().unwrap_or(0.0)));
                    ui.weak(r["workspace_dir"].as_str().unwrap_or(""));
                    ui.end_row();
                }
            });
        });
    }

    fn ui_logs(&mut self, ui: &mut egui::Ui) {
        ui.heading("Logs");
        ui.weak("live trainer output (SSE) — full log stays in the run workspace file");
        egui::ScrollArea::vertical().auto_shrink([false; 2]).stick_to_bottom(true).show(ui, |ui| {
            for line in &self.logs {
                ui.monospace(line);
            }
        });
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        ui.heading("Settings");
        ui.horizontal(|ui| {
            ui.label("Server");
            ui.add(egui::TextEdit::singleline(&mut self.base_edit).desired_width(320.0));
            if ui.button("Apply").clicked() {
                *self.base.lock().unwrap() = self.base_edit.trim_end_matches('/').to_string();
                self.save_settings();
                self.presets.clear();
                self.overrides.clear();
                self.status_line = format!("switched to {}", self.base_edit);
            }
        });
        ui.weak("point this at the trainer box (e.g. http://trainbox:8188) — this app renders locally, training stays remote");
        ui.add_space(8.0);
        ui.label(format!("settings file: {}", settings_path()));
        ui.label(format!("textures cached: {} (cap {TEXTURE_CAP})", self.textures.len()));
    }

    // ------------------------------------------------------------ rail

    fn ui_rail(&mut self, ui: &mut egui::Ui) {
        ui.add_space(6.0);
        let (status, color) = match &self.live {
            Some(r) => match r["status"].as_str().unwrap_or("") {
                "running" => ("RUNNING", Color32::from_rgb(80, 200, 120)),
                "exited" => ("EXITED", Color32::from_rgb(120, 160, 255)),
                "failed" => ("FAILED", Color32::from_rgb(230, 90, 90)),
                "stopped" => ("STOPPED", Color32::from_rgb(230, 160, 60)),
                _ => ("IDLE", Color32::GRAY),
            },
            None => {
                if self.active_run_id().is_some() {
                    ("RUNNING", Color32::from_rgb(80, 200, 120))
                } else {
                    ("IDLE", Color32::GRAY)
                }
            }
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!(" {status} ")).background_color(color).color(Color32::BLACK).strong());
        });
        ui.add_space(4.0);
        // prefer live SSE run; fall back to the poller's runs row
        let run = self
            .live
            .clone()
            .or_else(|| self.runs.iter().find(|r| r["status"] == "running").cloned());
        if let Some(r) = &run {
            let step = r["step"].as_u64().unwrap_or(0);
            let total = r["total_steps"].as_u64().unwrap_or(0).max(1);
            ui.add(egui::ProgressBar::new(step as f32 / total as f32).text(format!("{step}/{total}")));
            egui::Grid::new("rail").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
                let rows: [(&str, String); 5] = [
                    ("Loss", format!("{:.5}", r["loss"].as_f64().unwrap_or(0.0))),
                    ("Grad", format!("{:.5}", r["grad_norm"].as_f64().unwrap_or(0.0))),
                    ("Speed", format!("{:.2} s/step", r["s_per_step"].as_f64().unwrap_or(0.0))),
                    ("ETA", r["eta"].as_str().unwrap_or("—").to_string()),
                    ("Backend", r["backend"].as_str().unwrap_or("—").to_string()),
                ];
                for (k, v) in rows {
                    ui.weak(k);
                    ui.label(v);
                    ui.end_row();
                }
            });
        } else {
            ui.weak("no active run");
        }
        ui.add_space(6.0);
        chart(ui, "loss", &self.loss_series, Color32::from_rgb(90, 170, 255), 90.0);
        ui.add_space(4.0);
        chart(ui, "grad_norm", &self.grad_series, Color32::from_rgb(255, 150, 90), 70.0);
        ui.add_space(8.0);
        ui.separator();
        let m = &self.metrics;
        ui.strong("Hardware");
        egui::Grid::new("hw").num_columns(2).spacing([8.0, 3.0]).show(ui, |ui| {
            let rows: [(&str, String); 5] = [
                ("GPU", m["gpu_name"].as_str().unwrap_or("—").to_string()),
                ("Util", format!("{}%  {}°C", m["gpu_util"].as_str().unwrap_or("—"), m["gpu_temp"].as_str().unwrap_or("—"))),
                (
                    "VRAM",
                    format!(
                        "{}/{} MiB",
                        m["vram_used_mb"].as_str().unwrap_or("—"),
                        m["vram_total_mb"].as_str().unwrap_or("—")
                    ),
                ),
                (
                    "RAM",
                    format!(
                        "{}/{} GB",
                        m["ram_used_gb"].as_str().unwrap_or("—"),
                        m["ram_total_gb"].as_str().unwrap_or("—")
                    ),
                ),
                ("Driver", m["gpu_driver"].as_str().unwrap_or("—").to_string()),
            ];
            for (k, v) in rows {
                ui.weak(k);
                ui.add(egui::Label::new(v).truncate());
                ui.end_row();
            }
        });
    }
}

// ---------------------------------------------------------------- helpers

fn urlenc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'/' => out.push(b as char),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// Board scalars come back as an array; accept [x,y], [step,wall,value], or {x,y}.
fn parse_series(v: &Value) -> Vec<[f64; 2]> {
    let Some(arr) = v.as_array() else { return vec![] };
    arr.iter()
        .filter_map(|e| {
            if let Some(pair) = e.as_array() {
                let x = pair.first()?.as_f64()?;
                let y = pair.last()?.as_f64()?;
                Some([x, y])
            } else if e.is_object() {
                let x = e["x"].as_f64().or_else(|| e["step"].as_f64())?;
                let y = e["y"].as_f64().or_else(|| e["value"].as_f64())?;
                Some([x, y])
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------- eframe

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.drain(ctx);

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.strong("Serenity Trainer");
                ui.weak("native egui · supervisor API");
                ui.separator();
                ui.label(&self.status_line);
            });
        });

        egui::SidePanel::left("nav").resizable(false).default_width(130.0).show(ctx, |ui| {
            ui.add_space(8.0);
            for (sec, label) in SECTIONS {
                if ui.selectable_label(self.section == *sec, *label).clicked() {
                    self.section = *sec;
                }
            }
        });

        egui::SidePanel::right("rail").resizable(true).default_width(270.0).show(ctx, |ui| {
            self.ui_rail(ui);
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.section {
            Section::Train => self.ui_train(ui),
            Section::Dataset => self.ui_dataset(ui),
            Section::Captioner => self.ui_captioner(ui),
            Section::Validations => self.ui_validations(ui),
            Section::Samples => self.ui_samples(ui),
            Section::Board => self.ui_board(ui),
            Section::Runs => self.ui_runs(ui),
            Section::Logs => self.ui_logs(ui),
            Section::Settings => self.ui_settings(ui),
        });

        // lightbox overlay
        if let Some(url) = self.lightbox.clone() {
            let mut open = true;
            egui::Window::new("preview")
                .collapsible(false)
                .resizable(true)
                .open(&mut open)
                .default_size([900.0, 700.0])
                .show(ctx, |ui| {
                    if let Some(tex) = self.textures.get(&url) {
                        ui.add(egui::Image::new(tex).max_size(ui.available_size()));
                    }
                    ui.weak(&url);
                });
            if !open {
                self.lightbox = None;
            }
        }

        // keep the poller's updates visible even when idle
        ctx.request_repaint_after(Duration::from_millis(750));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.save_settings();
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        // logical points: the desktop may run fractional scaling (e.g. 1.75x on a
        // 2560-wide panel = ~1463 logical width), so stay under that.
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1240.0, 760.0])
            .with_position([20.0, 40.0])
            .with_title("Serenity Trainer (egui)"),
        ..Default::default()
    };
    eframe::run_native(
        "serenity-eguitrainer",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}
