"""Config list surface shared by Serenity list widgets."""

from mojoui.core.context import Context
from mojoui.widgets.form import begin_form_panel, end_form_panel, field_row
from serenity_trainer.ui.AuxScreenCommon import aux_label_w, aux_panel_h
from serenity_trainer.ui.UITabCommon import row1, value_w


def render_config_list(mut ctx: Context, title: String, content_w: Int32, rows: List[String]) raises:
    var label_w = aux_label_w(ctx, content_w)
    var val_w = value_w(content_w - ctx.theme.padding * 2, label_w)
    var row_count = Int32(len(rows))
    if row_count < 1:
        row_count = 1
    ctx.layout_row(row1(content_w), aux_panel_h(ctx, row_count))
    begin_form_panel(ctx, title.copy(), String("Add, clone, remove, and edit entries"))
    if len(rows) == 0:
        field_row(ctx, label_w, val_w, String("Empty"), String("No entries"))
    for i in range(len(rows)):
        field_row(ctx, label_w, val_w, String("#") + String(i + 1), rows[i].copy())
    end_form_panel(ctx)
