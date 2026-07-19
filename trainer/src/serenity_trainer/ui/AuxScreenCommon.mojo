"""Shared helpers for auxiliary Serenity UI surfaces."""

from mojoui.core.context import Context


def aux_panel_h(ctx: Context, rows: Int32) -> Int32:
    var pad = ctx.theme.padding
    var header_h = pad * 3
    var text_header_h = ctx.theme.font_size_pt * 3
    if text_header_h > header_h:
        header_h = text_header_h
    var gaps = rows - 1
    if gaps < 0:
        gaps = 0
    return header_h + pad * 2 + ctx.theme.row_height * rows + ctx.theme.spacing * gaps


def aux_label_w(ctx: Context, panel_w: Int32) -> Int32:
    var inner_w = panel_w - ctx.theme.padding * 2
    var w = ctx.theme.font_size_pt * 8
    if w < 178:
        w = 178
    var max_w = inner_w - 196
    if max_w < 132:
        max_w = 132
    if w > max_w:
        w = max_w
    return w
