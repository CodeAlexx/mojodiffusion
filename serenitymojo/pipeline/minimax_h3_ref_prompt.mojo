# serenitymojo/pipeline/minimax_h3_ref_prompt.mojo — MiniMax-H3 ref2va UNIT C:
# the structured prompt builder. Pure host string work: no Tensor, no
# DeviceContext, no GPU, no tokenizer.
#
# ── THE FORMAT (measured from the vendor's own request, not invented) ────────
# MiniMax-H3's ref2va task takes ONE prompt string carrying six labelled
# sections, in a fixed order. Measured against
# `creator-MiniMax-H3/scripts/readme/reproducible-768p-ref2va-request.sh` (its
# `.prompt` field, 3554 bytes):
#
#     subject_definitions:
#     <line>...
#     <BLANK>
#     summary:
#     <line>
#     <BLANK>
#     retention_analysis:
#     <line>...
#     <BLANK>
#     detailed_description:
#     <line>...
#     <BLANK>
#     overall_soundscape:
#     <line>
#     <BLANK>
#     non_diegetic_music:
#     <line>
#
# Precisely: each section is a header line `name:` followed by its body lines,
# body lines joined with a single `\n`, sections joined with `\n\n`, and NO
# trailing newline on the whole prompt (verified: the vendor's prompt does not
# end in `\n`). `non_diegetic_music` is the only optional section — leave it
# empty and it is omitted entirely, separator and all.
#
# The prompt is appended to the reference presentation VERBATIM, "with no chat
# template and no special tokens" (`build_ref2va_presentation`,
# packing_ref2va.py:756-819), so every byte here reaches the conditioner exactly
# as written. That is why this builder is byte-exact rather than approximately
# formatted, and why its probe compares against the vendor's own file.
#
# ── THE CROSS-REFERENCE LABELS ───────────────────────────────────────────────
# `<Subject i>`, `<Picture i>`, `<Audio j>`, `<Video k>` and `[Shot n]` are the
# prompt's own vocabulary. The media labels are NOT free-form: `<Picture i>` /
# `<Audio j>` / `<Video k>` are numbered per modality in PACKED ORDER by
# `build_ref2va_presentation` (:802-817), so a prompt that calls a reference
# `<Video 2>` while it is packed first is describing a different reference than
# the model sees. This module does not renumber them — it emits what it is
# given — so the caller must derive the labels from the same ordered reference
# list it hands to `models/dit/minimax_h3_ref_geometry.mojo`.
#
# ── DIALOGUE ─────────────────────────────────────────────────────────────────
# Spoken lines are wrapped in the model's dialogue markers,
# `<d>[English] ...</d>`, and go INSIDE `detailed_description`, inline in the
# shot prose — not in a section of their own. `minimax_h3_dialogue` builds one.
# `<d>` and `</d>` are among the seven special tokens the H3 tokenizer carries,
# so the wrapper must be exact.

from std.collections import List


# Section headers, in the order the vendor emits them. The trailing colon is
# part of the header line.
comptime MINIMAX_H3_SECTION_SUBJECT_DEFINITIONS = "subject_definitions"
comptime MINIMAX_H3_SECTION_SUMMARY = "summary"
comptime MINIMAX_H3_SECTION_RETENTION_ANALYSIS = "retention_analysis"
comptime MINIMAX_H3_SECTION_DETAILED_DESCRIPTION = "detailed_description"
comptime MINIMAX_H3_SECTION_OVERALL_SOUNDSCAPE = "overall_soundscape"
comptime MINIMAX_H3_SECTION_NON_DIEGETIC_MUSIC = "non_diegetic_music"

# The retention statuses the vendor's own examples use. The field is a free
# String, not an enum: these are the observed vocabulary, not a closed set the
# checkpoint is known to enforce.
comptime MINIMAX_H3_RETENTION_FULLY_PRESERVED = "fully_preserved"
comptime MINIMAX_H3_RETENTION_PARTIALLY_COPY = "partially_copy"
comptime MINIMAX_H3_RETENTION_REFERENCE = "reference"

comptime MINIMAX_H3_DIALOGUE_OPEN = "<d>"
comptime MINIMAX_H3_DIALOGUE_CLOSE = "</d>"


def minimax_h3_dialogue(language: String, text: String) raises -> String:
    """One spoken line: `<d>[English] Follow the wind, live free.</d>`.

    Goes inline inside `detailed_description`, in the shot prose. `<d>` and
    `</d>` are H3 tokenizer special tokens, so this wrapper is exact."""
    if language == String(""):
        raise Error("minimax_h3_ref_prompt: dialogue needs a language tag")
    return (
        String(MINIMAX_H3_DIALOGUE_OPEN) + "[" + language + "] " + text
        + String(MINIMAX_H3_DIALOGUE_CLOSE)
    )


@fieldwise_init
struct MiniMaxH3SubjectDefinition(Copyable, Movable):
    """One `subject_definitions` line: `<label> is <description>`.

    `label` is the cross-reference token (`<Subject 1>`, `<Video 1>`,
    `<Audio 2>`); `description` carries its own terminating punctuation, which
    the vendor's lines do."""

    var label: String
    var description: String

    def render(self) raises -> String:
        if self.label == String("") or self.description == String(""):
            raise Error(
                "minimax_h3_ref_prompt: a subject definition needs a label and"
                " a description"
            )
        return self.label + String(" is ") + self.description


@fieldwise_init
struct MiniMaxH3RetentionEntry(Copyable, Movable):
    """One `retention_analysis` line.

    Rendered `<label> (<qualifier>): <status> - <description>`, with the
    parenthesised qualifier omitted when empty — the vendor writes
    `<Subject 1> (appears in [Shot 1]): fully_preserved - ...` for a subject
    but a bare `<Audio 1>: partially_copy - ...` for a soundtrack."""

    var label: String
    var qualifier: String
    var status: String
    var description: String

    def render(self) raises -> String:
        if self.label == String("") or self.status == String(""):
            raise Error(
                "minimax_h3_ref_prompt: a retention entry needs a label and a"
                " status"
            )
        var out = self.label
        if self.qualifier != String(""):
            out += String(" (") + self.qualifier + String(")")
        out += String(": ") + self.status
        if self.description != String(""):
            out += String(" - ") + self.description
        return out^


@fieldwise_init
struct MiniMaxH3Shot(Copyable, Movable):
    """One `detailed_description` shot line: `[Shot 1] <text>`."""

    var label: String
    var text: String

    def render(self) raises -> String:
        if self.label == String("") or self.text == String(""):
            raise Error("minimax_h3_ref_prompt: a shot needs a label and text")
        return self.label + String(" ") + self.text


def minimax_h3_summary_line(tags: List[String], text: String) raises -> String:
    """The `summary` body: an optional bracketed task-tag list, then the prose.

    The vendor's example opens with
    `[video editing + audio reference + audio reuse] The target video is ...`,
    i.e. tags joined by ` + ` inside one bracket, then a single space, then the
    summary. With no tags the prose stands alone."""
    if text == String(""):
        raise Error("minimax_h3_ref_prompt: summary needs text")
    if len(tags) == 0:
        return text
    var joined = String("[")
    for i in range(len(tags)):
        if i > 0:
            joined += String(" + ")
        joined += tags[i]
    joined += String("] ") + text
    return joined^


def _render_section(name: String, body: List[String]) raises -> String:
    """`name:` then the body lines, joined with a single newline."""
    if len(body) == 0:
        raise Error(
            String("minimax_h3_ref_prompt: section '") + name + "' has no body"
        )
    var out = name + String(":")
    for i in range(len(body)):
        if body[i] == String(""):
            raise Error(
                String("minimax_h3_ref_prompt: section '") + name
                + "' has an empty line at index " + String(i)
                + " — a blank line would split the section, since sections are"
                " separated by a blank line"
            )
        out += String("\n") + body[i]
    return out^


@fieldwise_init
struct MiniMaxH3Ref2VAPrompt(Copyable, Movable):
    """The six ref2va sections, as already-rendered body lines.

    Held as `List[String]` of rendered lines rather than as the typed structs so
    that a caller may mix rendered helpers with hand-written lines — the vendor
    format is prose, and pinning every line to a struct would fight it. Use
    `MiniMaxH3SubjectDefinition` / `MiniMaxH3RetentionEntry` / `MiniMaxH3Shot`
    to build the regular ones and append their `render()`.

    `non_diegetic_music` is the only optional section: leave it empty to omit
    it, separator included."""

    var subject_definitions: List[String]
    var summary: String
    var retention_analysis: List[String]
    var detailed_description: List[String]
    var overall_soundscape: String
    var non_diegetic_music: String

    def render(self) raises -> String:
        """The full prompt: six sections joined by a blank line, no trailing
        newline."""
        var sections = List[String]()
        sections.append(
            _render_section(
                String(MINIMAX_H3_SECTION_SUBJECT_DEFINITIONS),
                self.subject_definitions,
            )
        )
        sections.append(
            _render_section(String(MINIMAX_H3_SECTION_SUMMARY), [self.summary])
        )
        sections.append(
            _render_section(
                String(MINIMAX_H3_SECTION_RETENTION_ANALYSIS),
                self.retention_analysis,
            )
        )
        sections.append(
            _render_section(
                String(MINIMAX_H3_SECTION_DETAILED_DESCRIPTION),
                self.detailed_description,
            )
        )
        sections.append(
            _render_section(
                String(MINIMAX_H3_SECTION_OVERALL_SOUNDSCAPE),
                [self.overall_soundscape],
            )
        )
        if self.non_diegetic_music != String(""):
            sections.append(
                _render_section(
                    String(MINIMAX_H3_SECTION_NON_DIEGETIC_MUSIC),
                    [self.non_diegetic_music],
                )
            )

        var out = String("")
        for i in range(len(sections)):
            if i > 0:
                out += String("\n\n")
            out += sections[i]
        return out^


# ═════════════════════════════════════════════════════════════════════════════
# THE BASE (T2VA / I2VA / FL2VA / L2VA) PROMPT — a DIFFERENT format from the
# six-section ref2va one above.
#
# Authority: creator-MiniMax-H3/docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md,
# §2.1 (:12-34) for the alignment instruction and §2.2 (:36-48) for the three
# core fields, with the four worked cases at :168-222 as the byte-level check.
#
# Structure (§2.1:34, verbatim): "The instruction must be the first line of the
# final prompt, followed by one blank line before the core fields." T2VA has no
# instruction and starts directly with the fields.
#
#   integrated_multimodal_description: [Shot 1] ...
#   <BLANK>
#   overall_soundscape: ...
#   <BLANK>
#   non_diegetic_music: ...
#
# Header and body share a line here, separated by ": " — unlike the ref2va
# format, where the header is a line of its own. Confirmed against all four
# cases (:174-222).
#
# ── THE TEMPLATES ARE COPIED, INCLUDING THEIR INCONSISTENCY ─────────────────
# I2VA and L2VA bracket their cross-references (`<Picture 1>`, `[Shot 1]`);
# FL2VA does NOT (`Picture 1`, `Shot 1`). That is not a typo in this file — it
# is what the guide prints (:19, :25, :31) and what its own worked cases repeat
# (:187, :201, :215). "Normalizing" it would be inventing a template the model
# was not shown. The separator in the FL2VA/L2VA templates is an EM DASH
# (U+2014), not a hyphen.
# ═════════════════════════════════════════════════════════════════════════════

comptime MINIMAX_H3_TASK_T2VA = 0
comptime MINIMAX_H3_TASK_I2VA = 1
comptime MINIMAX_H3_TASK_FL2VA = 2
comptime MINIMAX_H3_TASK_L2VA = 3

comptime MINIMAX_H3_FIELD_DESCRIPTION = "integrated_multimodal_description"
comptime MINIMAX_H3_FIELD_SOUNDSCAPE = "overall_soundscape"
comptime MINIMAX_H3_FIELD_MUSIC = "non_diegetic_music"

# MiniMax-H3's fixed frame rate — the divisor that turns a frame count into the
# `S.SS` the alignment instruction carries.
comptime MINIMAX_H3_PROMPT_FPS = 24


def minimax_h3_format_two_decimals(numerator: Int, denominator: Int) raises -> String:
    """Python's `"{:.2f}"` of an exact rational, computed in INTEGERS.

    The alignment instruction's `S.SS` is a duration, and a duration here is
    always `num_frames / 24` — a repeating decimal for most legal frame counts
    (124 frames is 5.1666...). Formatting it by rounding `value * 100.0` asks a
    double to decide a tie it cannot represent; doing the division in integers
    with a round-half-to-even on the remainder is exact, and the probe checks
    that result against CPython's own `"{:.2f}"` for EVERY legal frame count
    rather than trusting the argument."""
    if denominator <= 0:
        raise Error("minimax_h3_ref_prompt: denominator must be positive")
    if numerator < 0:
        raise Error("minimax_h3_ref_prompt: a duration cannot be negative")
    var scaled = numerator * 100
    var whole = scaled // denominator
    var rem = scaled % denominator
    # Round half to even on the exact remainder.
    var twice = rem * 2
    if twice > denominator:
        whole += 1
    elif twice == denominator:
        if whole % 2 != 0:
            whole += 1
    var units = whole // 100
    var hundredths = whole % 100
    var frac = String(hundredths)
    if hundredths < 10:
        frac = String("0") + frac
    return String(units) + "." + frac


def minimax_h3_alignment_seconds(num_frames: Int) raises -> String:
    """The `S.SS` of a request: its effective duration at MiniMax-H3's fixed
    24 fps. `num_frames` is the ALIGNED (17n+5) count, because that is the
    duration the request actually generates (before_encoder.py:94-104 makes the
    same point about the duration ceiling)."""
    if num_frames < 1:
        raise Error("minimax_h3_ref_prompt: num_frames must be positive")
    return minimax_h3_format_two_decimals(num_frames, MINIMAX_H3_PROMPT_FPS)


def minimax_h3_alignment_instruction(
    task: Int, num_frames: Int, final_shot_index: Int
) raises -> String:
    """§2.1's instruction line for a task, byte for byte.

    `final_shot_index` is the guide's `N`, "the index of the actual final shot"
    (:34) — it is a property of the prose the caller wrote, not of the geometry,
    so it is passed in rather than guessed at. T2VA returns an empty string:
    it "has no image-alignment instruction and begins directly with the three
    core fields" (:14)."""
    if task == MINIMAX_H3_TASK_T2VA:
        return String("")
    if final_shot_index < 1:
        raise Error(
            "minimax_h3_ref_prompt: the final shot index is 1-based and must be"
            " at least 1"
        )
    var seconds = minimax_h3_alignment_seconds(num_frames)
    var shot_n = String(final_shot_index)

    if task == MINIMAX_H3_TASK_I2VA:
        # :19 — no duration and no N: the first frame is always 0.00 s in Shot 1.
        return String(
            "For the target video, at 0.00 seconds into the target video,"
            " <Picture 1> (from [Shot 1]) is fully referenced."
        )
    if task == MINIMAX_H3_TASK_FL2VA:
        # :25 — UNBRACKETED labels, em dash, two clauses joined by "; ".
        return (
            String("How the reference pictures align with the target video — ")
            + "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the"
            " target video; Picture 2 (from Shot " + shot_n + ") aligns with the "
            + seconds + "-second mark of the target video."
        )
    if task == MINIMAX_H3_TASK_L2VA:
        # :31 — BRACKETED labels, one clause, the picture lands at the END.
        return (
            String("How the reference pictures align with the target video — ")
            + "<Picture 1> (from [Shot " + shot_n + "]) aligns with the "
            + seconds + "-second mark of the target video."
        )
    raise Error(
        String("minimax_h3_ref_prompt: unknown task ") + String(task)
        + " — expected T2VA, I2VA, FL2VA or L2VA"
    )


@fieldwise_init
struct MiniMaxH3BasePrompt(Copyable, Movable):
    """The three core fields of a t2va/keyframe request (§2.2).

    `non_diegetic_music` is written `N/A` rather than omitted when there is no
    score (§4.7:162) — unlike the ref2va format's optional last section. Same
    for `overall_soundscape` under explicit total silence (§4.6:154). So both
    are required strings here, and a caller that means "none" says `N/A`."""

    var integrated_multimodal_description: String
    var overall_soundscape: String
    var non_diegetic_music: String

    def render(self) raises -> String:
        if self.integrated_multimodal_description == String(""):
            raise Error(
                "minimax_h3_ref_prompt: integrated_multimodal_description is the"
                " body of the prompt and cannot be empty"
            )
        if self.overall_soundscape == String(""):
            raise Error(
                "minimax_h3_ref_prompt: overall_soundscape cannot be empty —"
                " write N/A for explicit silence"
            )
        if self.non_diegetic_music == String(""):
            raise Error(
                "minimax_h3_ref_prompt: non_diegetic_music cannot be empty —"
                " write N/A when there is no score"
            )
        return (
            String(MINIMAX_H3_FIELD_DESCRIPTION) + ": "
            + self.integrated_multimodal_description + "\n\n"
            + String(MINIMAX_H3_FIELD_SOUNDSCAPE) + ": "
            + self.overall_soundscape + "\n\n"
            + String(MINIMAX_H3_FIELD_MUSIC) + ": " + self.non_diegetic_music
        )


def minimax_h3_keyframe_prompt(
    task: Int,
    body: MiniMaxH3BasePrompt,
    num_frames: Int,
    final_shot_index: Int = 1,
) raises -> String:
    """The full prompt a keyframe request sends: the §2.1 instruction, one blank
    line, then the three core fields (§2.1:34). T2VA emits the fields alone."""
    var instruction = minimax_h3_alignment_instruction(
        task, num_frames, final_shot_index
    )
    var fields = body.render()
    if instruction == String(""):
        return fields^
    return instruction + "\n\n" + fields


def minimax_h3_task_from_keyframes(have_first: Bool, have_last: Bool) -> Int:
    """Which of the four tasks a request is, from which keyframe slots it filled
    — the same `(image, last_image)` pair `keyframe_anchors` reads
    (before_encoder.py:167-171)."""
    if have_first and have_last:
        return MINIMAX_H3_TASK_FL2VA
    if have_first:
        return MINIMAX_H3_TASK_I2VA
    if have_last:
        return MINIMAX_H3_TASK_L2VA
    return MINIMAX_H3_TASK_T2VA


def minimax_h3_ref2va_section_names() -> List[String]:
    """The six section names, in emission order — what a parser or a gate
    checks a prompt's headers against."""
    return [
        String(MINIMAX_H3_SECTION_SUBJECT_DEFINITIONS),
        String(MINIMAX_H3_SECTION_SUMMARY),
        String(MINIMAX_H3_SECTION_RETENTION_ANALYSIS),
        String(MINIMAX_H3_SECTION_DETAILED_DESCRIPTION),
        String(MINIMAX_H3_SECTION_OVERALL_SOUNDSCAPE),
        String(MINIMAX_H3_SECTION_NON_DIEGETIC_MUSIC),
    ]
