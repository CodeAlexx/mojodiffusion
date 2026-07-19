# ConceptConfig.mojo — read a Serenity concept JSON (the dataset definition).
#
# 1:1 port of modules/util/config/ConceptConfig.py (ConceptImageConfig L9-81,
# ConceptTextConfig L84-125, ConceptConfig L128-200). The concept file is a JSON
# LIST of concept objects, each with nested "image" + "text" sub-objects.
#
# REUSES the parser machinery from util/config/TrainConfigReader.mojo (cursor +
# scalar readers + file read) — no second parser. Adds top-level ARRAY parsing
# and nested-object parsing for image/text (like TrainConfigReader._parse_optimizer).
# Start from defaults and overwrite each field as its key is parsed; unknown keys
# skipped, missing keys keep defaults. Mojo 1.0.0b1; no Python.

from std.collections import List
from serenitymojo.io.json_header import _Cursor, _parse_string, _skip_value
from serenity_trainer.util.config.TrainConfigReader import (
    _read_scalar, _read_file_bytes,
)


# ── ConceptImageConfig (ConceptConfig.py:9-81) ───────────────────────────────
@fieldwise_init
struct ConceptImageConfig(Copyable, Movable):
    var enable_crop_jitter: Bool                 # :49 default True
    var enable_random_flip: Bool                 # :51
    var enable_fixed_flip: Bool                  # :52
    var enable_resolution_override: Bool         # :74
    var resolution_override: String              # :75 default "512"

    @staticmethod
    def default() -> ConceptImageConfig:
        return ConceptImageConfig(True, False, False, False, String("512"))


# ── ConceptTextConfig (ConceptConfig.py:84-125) ──────────────────────────────
@fieldwise_init
struct ConceptTextConfig(Copyable, Movable):
    var prompt_source: String          # :109 default "sample"
    var prompt_path: String            # :110
    var enable_tag_shuffling: Bool     # :111

    @staticmethod
    def default() -> ConceptTextConfig:
        return ConceptTextConfig(String("sample"), String(""), False)


# ── ConceptConfig (ConceptConfig.py:128-200) ─────────────────────────────────
struct ConceptConfig(Copyable, Movable):
    var name: String                   # :187
    var path: String                   # :188 (image dir)
    var seed: Int                      # :189
    var enabled: Bool                  # :190 default True
    var concept_type: String           # :191 ConceptType (STANDARD/VALIDATION/PRIOR_PREDICTION)
    var include_subdirectories: Bool   # :192
    var image_variations: Int          # :193 default 1
    var text_variations: Int           # :194 default 1
    var balancing: Float32             # :195 default 1.0 (was "repeats", migration_0)
    var loss_weight: Float32           # :197 default 1.0
    var image: ConceptImageConfig
    var text: ConceptTextConfig

    def __init__(out self):
        self.name = String("")
        self.path = String("")
        self.seed = 0
        self.enabled = True
        self.concept_type = String("STANDARD")
        self.include_subdirectories = False
        self.image_variations = 1
        self.text_variations = 1
        self.balancing = Float32(1.0)
        self.loss_weight = Float32(1.0)
        self.image = ConceptImageConfig.default()
        self.text = ConceptTextConfig.default()

    def __init__(out self, *, copy: Self):
        self.name = copy.name
        self.path = copy.path
        self.seed = copy.seed
        self.enabled = copy.enabled
        self.concept_type = copy.concept_type
        self.include_subdirectories = copy.include_subdirectories
        self.image_variations = copy.image_variations
        self.text_variations = copy.text_variations
        self.balancing = copy.balancing
        self.loss_weight = copy.loss_weight
        self.image = copy.image.copy()
        self.text = copy.text.copy()


# Parse the nested "image" object into cfg.image (ConceptImageConfig.py).
def _parse_image(mut cur: _Cursor, mut cfg: ConceptConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance(); return
    while True:
        var f = _parse_string(cur)
        cur.expect(0x3A)
        if f == "enable_crop_jitter":
            cfg.image.enable_crop_jitter = _read_scalar(cur).num != 0.0
        elif f == "enable_random_flip":
            cfg.image.enable_random_flip = _read_scalar(cur).num != 0.0
        elif f == "enable_fixed_flip":
            cfg.image.enable_fixed_flip = _read_scalar(cur).num != 0.0
        elif f == "enable_resolution_override":
            cfg.image.enable_resolution_override = _read_scalar(cur).num != 0.0
        elif f == "resolution_override":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.image.resolution_override = sc.s
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C: cur.advance(); continue
        if ch == 0x7D: cur.advance(); break
        raise Error(String("concept JSON: bad image obj at byte ") + String(cur.pos))


# Parse the nested "text" object into cfg.text (ConceptTextConfig.py).
def _parse_text(mut cur: _Cursor, mut cfg: ConceptConfig) raises:
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance(); return
    while True:
        var f = _parse_string(cur)
        cur.expect(0x3A)
        if f == "prompt_source":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.text.prompt_source = sc.s
        elif f == "prompt_path":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.text.prompt_path = sc.s
        elif f == "enable_tag_shuffling":
            cfg.text.enable_tag_shuffling = _read_scalar(cur).num != 0.0
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C: cur.advance(); continue
        if ch == 0x7D: cur.advance(); break
        raise Error(String("concept JSON: bad text obj at byte ") + String(cur.pos))


# Parse one concept object (ConceptConfig.default_values L180-200).
def _parse_concept(mut cur: _Cursor) raises -> ConceptConfig:
    var cfg = ConceptConfig()
    cur.expect(0x7B)
    cur.skip_ws()
    if cur.peek() == 0x7D:
        cur.advance(); return cfg^
    while True:
        var key = _parse_string(cur)
        cur.expect(0x3A)
        if key == "name":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.name = sc.s
        elif key == "path":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.path = sc.s
        elif key == "seed":
            cfg.seed = Int(_read_scalar(cur).num)
        elif key == "enabled":
            cfg.enabled = _read_scalar(cur).num != 0.0
        elif key == "type":
            var sc = _read_scalar(cur)
            if sc.is_string: cfg.concept_type = sc.s
        elif key == "include_subdirectories":
            cfg.include_subdirectories = _read_scalar(cur).num != 0.0
        elif key == "image_variations":
            cfg.image_variations = Int(_read_scalar(cur).num)
        elif key == "text_variations":
            cfg.text_variations = Int(_read_scalar(cur).num)
        elif key == "balancing" or key == "repeats":   # migration_0: repeats→balancing (:157-158)
            cfg.balancing = Float32(_read_scalar(cur).num)
        elif key == "loss_weight":
            cfg.loss_weight = Float32(_read_scalar(cur).num)
        elif key == "image":
            _parse_image(cur, cfg)
        elif key == "text":
            _parse_text(cur, cfg)
        else:
            _skip_value(cur)
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C: cur.advance(); continue
        if ch == 0x7D: cur.advance(); break
        raise Error(String("concept JSON: bad concept obj at byte ") + String(cur.pos))
    return cfg^


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: read the concept JSON (a top-level ARRAY) into a list of ConceptConfig.
# ─────────────────────────────────────────────────────────────────────────────
def read_concepts(json_path: String) raises -> List[ConceptConfig]:
    var bytes = _read_file_bytes(json_path)
    var cur = _Cursor(bytes^)
    var out = List[ConceptConfig]()

    cur.skip_ws()
    cur.expect(0x5B)  # top-level '['
    cur.skip_ws()
    if cur.peek() == 0x5D:  # empty array
        cur.advance()
        return out^

    while True:
        out.append(_parse_concept(cur))
        cur.skip_ws()
        var ch = cur.peek()
        if ch == 0x2C: cur.advance(); cur.skip_ws(); continue
        if ch == 0x5D: cur.advance(); break
        raise Error(String("concept JSON: expected ',' or ']' at byte ") + String(cur.pos))

    return out^
