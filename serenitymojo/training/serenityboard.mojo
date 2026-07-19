# Pure-Mojo SerenityBoard SQLite writer for trainers.
#
# Runtime trainers use this directly. Python remains useful for parity tests,
# but trainer progress/UI/board emission must not depend on Python.

from std.ffi import external_call
from std.memory import alloc
from std.time import perf_counter_ns

from serenitymojo.io.ffi import (
    BytePtr,
    sys_open,
    sys_close,
    sys_pread,
    sys_pwrite,
    sys_system,
    O_RDONLY,
    O_WRONLY,
    O_CREAT,
    O_TRUNC,
)


def _null() -> BytePtr:
    return BytePtr(unsafe_from_address=0)


def _cstring(s: String) -> BytePtr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return BytePtr(unsafe_from_address=Int(buf))


def _free_cstring(p: BytePtr):
    p.free()


def _sql_quote(s: String) -> String:
    var out = String("'")
    for ch in s.codepoint_slices():
        var c = String(ch)
        if c == String("'"):
            out += String("''")
        else:
            out += c
    out += String("'")
    return out^


def _db(db_addr: Int) -> BytePtr:
    return BytePtr(unsafe_from_address=db_addr)


def _sqlite_open(path: String) raises -> Int:
    var out = alloc[Int](1)
    out[0] = 0
    var cpath = _cstring(path)
    var rc = Int(external_call["sqlite3_open", Int32](
        cpath, BytePtr(unsafe_from_address=Int(out))
    ))
    _free_cstring(cpath)
    var db_addr = out[0]
    out.free()
    if rc != 0 or db_addr == 0:
        raise Error(String("serenityboard: sqlite3_open failed for ") + path)
    return db_addr


def _sqlite_close(db_addr: Int):
    if db_addr != 0:
        _ = external_call["sqlite3_close", Int32](_db(db_addr))


def _sqlite_exec(db_addr: Int, sql: String) raises:
    var csql = _cstring(sql)
    var rc = Int(external_call["sqlite3_exec", Int32](
        _db(db_addr), csql, _null(), _null(), _null()
    ))
    _free_cstring(csql)
    if rc != 0:
        raise Error(String("serenityboard: sqlite exec failed rc=") + String(rc))


def _now_seconds() -> Float64:
    # Monotonic seconds are enough for ordering; the UI keys by step/tag.
    return Float64(perf_counter_ns()) / 1.0e9


def _schema_sql() -> String:
    return (
        String("PRAGMA journal_mode = WAL;")
        + String("PRAGMA synchronous = NORMAL;")
        + String("PRAGMA busy_timeout = 5000;")
        + String("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);")
        + String("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT NOT NULL, start_time REAL NOT NULL, resume_step INTEGER, status TEXT NOT NULL CHECK(status IN ('running','complete','crashed')), PRIMARY KEY (session_id)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS scalars (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, value REAL NOT NULL, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS tensors (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, dtype TEXT NOT NULL, shape TEXT NOT NULL, data BLOB NOT NULL, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS artifacts (tag TEXT NOT NULL, step INTEGER NOT NULL, seq_index INTEGER NOT NULL DEFAULT 0, wall_time REAL NOT NULL, kind TEXT NOT NULL, mime_type TEXT NOT NULL, blob_key TEXT NOT NULL, width INTEGER, height INTEGER, meta TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (tag, step, seq_index)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS text_events (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, value TEXT NOT NULL, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS trace_events (step INTEGER NOT NULL, wall_time REAL NOT NULL, phase TEXT NOT NULL, duration_ms REAL NOT NULL, details TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (step, phase)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS eval_results (suite_name TEXT NOT NULL, case_id TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, score_name TEXT NOT NULL, score_value REAL NOT NULL, artifact_key TEXT, details TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (suite_name, case_id, step, score_name)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS hparam_metrics (metric_tag TEXT NOT NULL, value REAL NOT NULL, step INTEGER, wall_time REAL, PRIMARY KEY (metric_tag)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS plugin_data (plugin_name TEXT NOT NULL, tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, data TEXT NOT NULL, PRIMARY KEY (plugin_name, tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS custom_scalar_layouts (layout_name TEXT NOT NULL, config TEXT NOT NULL, PRIMARY KEY (layout_name)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS pr_curves (tag TEXT NOT NULL, step INTEGER NOT NULL, class_index INTEGER NOT NULL DEFAULT 0, wall_time REAL NOT NULL, num_thresholds INTEGER NOT NULL, data BLOB NOT NULL, PRIMARY KEY (tag, step, class_index)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS audio (tag TEXT NOT NULL, step INTEGER NOT NULL, seq_index INTEGER NOT NULL DEFAULT 0, wall_time REAL NOT NULL, blob_key TEXT NOT NULL, sample_rate INTEGER NOT NULL, num_channels INTEGER NOT NULL DEFAULT 1, duration_ms REAL, mime_type TEXT NOT NULL DEFAULT 'audio/wav', label TEXT NOT NULL DEFAULT '', PRIMARY KEY (tag, step, seq_index)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS graphs (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, graph_blob_key TEXT NOT NULL, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS embeddings (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, num_points INTEGER NOT NULL, dimensions INTEGER NOT NULL, tensor_blob_key TEXT NOT NULL, metadata_json TEXT, metadata_header TEXT, sprite_blob_key TEXT, sprite_single_h INTEGER, sprite_single_w INTEGER, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE TABLE IF NOT EXISTS meshes (tag TEXT NOT NULL, step INTEGER NOT NULL, wall_time REAL NOT NULL, num_vertices INTEGER NOT NULL, has_faces INTEGER NOT NULL DEFAULT 0, has_colors INTEGER NOT NULL DEFAULT 0, num_faces INTEGER NOT NULL DEFAULT 0, vertices_blob_key TEXT NOT NULL, faces_blob_key TEXT, colors_blob_key TEXT, config_json TEXT, PRIMARY KEY (tag, step)) WITHOUT ROWID;")
        + String("CREATE INDEX IF NOT EXISTS idx_scalars_tag ON scalars(tag);")
        + String("CREATE INDEX IF NOT EXISTS idx_scalars_tag_step ON scalars(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_tensors_tag_step ON tensors(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_artifacts_tag_step ON artifacts(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_text_tag_step ON text_events(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_eval_suite_step ON eval_results(suite_name, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_plugin_name_tag ON plugin_data(plugin_name, tag);")
        + String("CREATE INDEX IF NOT EXISTS idx_pr_curves_tag_step ON pr_curves(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_audio_tag_step ON audio(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_graphs_tag_step ON graphs(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_embeddings_tag_step ON embeddings(tag, step DESC);")
        + String("CREATE INDEX IF NOT EXISTS idx_meshes_tag_step ON meshes(tag, step DESC);")
    )


def _copy_file(src: String, dst: String) raises:
    var in_fd = sys_open(src, O_RDONLY, 0)
    if in_fd < 0:
        raise Error(String("serenityboard: cannot read artifact ") + src)
    var out_fd = sys_open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
    if out_fd < 0:
        _ = sys_close(in_fd)
        raise Error(String("serenityboard: cannot write artifact ") + dst)
    comptime CHUNK = 1048576
    var buf = alloc[UInt8](CHUNK)
    var off = 0
    while True:
        var n = sys_pread(in_fd, BytePtr(unsafe_from_address=Int(buf)), CHUNK, off)
        if n < 0:
            buf.free()
            _ = sys_close(in_fd)
            _ = sys_close(out_fd)
            raise Error("serenityboard: artifact read error")
        if n == 0:
            break
        var w = sys_pwrite(out_fd, BytePtr(unsafe_from_address=Int(buf)), n, off)
        if w != n:
            buf.free()
            _ = sys_close(in_fd)
            _ = sys_close(out_fd)
            raise Error("serenityboard: artifact write error")
        off += n
        if n < CHUNK:
            break
    buf.free()
    _ = sys_close(in_fd)
    _ = sys_close(out_fd)


def _png_dims(path: String) -> Tuple[Int, Int]:
    var fd = sys_open(path, O_RDONLY, 0)
    if fd < 0:
        return (0, 0)
    var buf = alloc[UInt8](24)
    var n = sys_pread(fd, BytePtr(unsafe_from_address=Int(buf)), 24, 0)
    _ = sys_close(fd)
    if n < 24 or buf[1] != 0x50 or buf[2] != 0x4E or buf[3] != 0x47:
        buf.free()
        return (0, 0)
    var w = (Int(buf[16]) << 24) | (Int(buf[17]) << 16) | (Int(buf[18]) << 8) | Int(buf[19])
    var h = (Int(buf[20]) << 24) | (Int(buf[21]) << 16) | (Int(buf[22]) << 8) | Int(buf[23])
    buf.free()
    return (w, h)


@fieldwise_init
struct SerenityBoardWriter(Movable):
    var db_addr: Int
    var output_dir: String
    var session_id: String

    @staticmethod
    def open(output_dir: String, session_id: String, resume_step: Int) raises -> SerenityBoardWriter:
        _ = sys_system(String("mkdir -p ") + output_dir)
        _ = sys_system(String("mkdir -p ") + output_dir + String("/blobs"))
        var db_path = output_dir + String("/board.db")
        var db_addr = _sqlite_open(db_path)
        _sqlite_exec(db_addr, _schema_sql())
        var now = _now_seconds()
        _sqlite_exec(
            db_addr,
            String("INSERT OR REPLACE INTO sessions (session_id, start_time, resume_step, status) VALUES (")
            + _sql_quote(session_id) + String(", ") + String(now) + String(", ")
            + String(resume_step) + String(", 'running');")
        )
        _sqlite_exec(
            db_addr,
            String("INSERT OR REPLACE INTO metadata (key, value) VALUES ('active_session_id', ")
            + _sql_quote(String("\"") + session_id + String("\"")) + String(");")
        )
        _sqlite_exec(db_addr, String("INSERT OR REPLACE INTO metadata (key, value) VALUES ('name', '\"Klein-lora\"');"))
        _sqlite_exec(db_addr, String("INSERT OR REPLACE INTO metadata (key, value) VALUES ('status', '\"running\"');"))
        return SerenityBoardWriter(db_addr, output_dir.copy(), session_id.copy())

    def close(mut self):
        _sqlite_close(self.db_addr)
        self.db_addr = 0

    def set_status(self, status: String) raises:
        var canonical = String("running")
        if status == String("complete") or status == String("completed"):
            canonical = String("complete")
        elif status == String("crashed") or status == String("failed") or status == String("error"):
            canonical = String("crashed")
        _sqlite_exec(
            self.db_addr,
            String("UPDATE sessions SET status = ") + _sql_quote(canonical)
            + String(" WHERE session_id = ") + _sql_quote(self.session_id) + String(";")
        )
        _sqlite_exec(
            self.db_addr,
            String("INSERT OR REPLACE INTO metadata (key, value) VALUES ('status', ")
            + _sql_quote(String("\"") + canonical + String("\"")) + String(");")
        )

    def log_scalar(self, tag: String, step: Int, value: Float64) raises:
        _sqlite_exec(
            self.db_addr,
            String("INSERT OR REPLACE INTO scalars (tag, step, wall_time, value) VALUES (")
            + _sql_quote(tag) + String(", ") + String(step) + String(", ")
            + String(_now_seconds()) + String(", ") + String(value) + String(");")
        )

    def log_train_step(
        self,
        step: Int,
        loss: Float32,
        grad_norm: Float64,
        lr: Float32,
        step_secs: Float64,
        noise_speed: Float64,
    ) raises:
        self.log_scalar(String("loss/train"), step, Float64(loss))
        self.log_scalar(String("grad_norm"), step, grad_norm)
        self.log_scalar(String("lr/default"), step, Float64(lr))
        var sps = Float64(0.0)
        if step_secs > 0.0:
            sps = 1.0 / step_secs
        self.log_scalar(String("perf/steps_per_sec"), step, sps)
        self.log_scalar(String("perf/sec_per_step"), step, step_secs)
        self.log_scalar(String("perf/noise_elems_per_sec"), step, noise_speed)

    def log_text(self, tag: String, step: Int, value: String) raises:
        _sqlite_exec(
            self.db_addr,
            String("INSERT OR REPLACE INTO text_events (tag, step, wall_time, value) VALUES (")
            + _sql_quote(tag) + String(", ") + String(step) + String(", ")
            + String(_now_seconds()) + String(", ") + _sql_quote(value) + String(");")
        )

    def log_hparams(self, hparams_json: String) raises:
        _sqlite_exec(
            self.db_addr,
            String("INSERT OR REPLACE INTO metadata (key, value) VALUES ('hparams', ")
            + _sql_quote(hparams_json) + String(");")
        )

    def log_image_png(self, tag: String, step: Int, seq_index: Int, png_path: String) raises:
        var blob_key = String("sample_step") + String(step) + String("_") + String(seq_index) + String(".png")
        var dst = self.output_dir + String("/blobs/") + blob_key
        _copy_file(png_path, dst)
        var dims = _png_dims(png_path)
        _sqlite_exec(
            self.db_addr,
            String("INSERT OR REPLACE INTO artifacts (tag, step, seq_index, wall_time, kind, mime_type, blob_key, width, height, meta) VALUES (")
            + _sql_quote(tag) + String(", ") + String(step) + String(", ") + String(seq_index)
            + String(", ") + String(_now_seconds()) + String(", 'image', 'image/png', ")
            + _sql_quote(blob_key) + String(", ") + String(dims[0]) + String(", ")
            + String(dims[1]) + String(", '{}');")
        )
