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
