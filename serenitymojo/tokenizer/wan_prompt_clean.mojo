# Wan creator prompt cleanup used before the UMT5 tokenizer.
#
# The creator applies `ftfy.fix_text`, HTML unescape, and whitespace cleanup in
# `wan/modules/tokenizers.py`.  The quality-critical default negative prompt
# contains full-width punctuation; ftfy folds that punctuation to ASCII before
# tokenization.  UMT5 otherwise produces different token IDs and conditioning.

from serenitymojo.tokenizer.tokenizer import (
    _str_to_cps,
    _cps_to_string,
    is_whitespace,
)


def wan_creator_clean(text: String) -> String:
    """Apply creator-compatible width folding and whitespace cleanup.

    Full-width ASCII U+FF01..U+FF5E maps to U+0021..U+007E, matching the
    normalization `ftfy.fix_text` applies to Wan's bundled negative prompt.
    Whitespace runs collapse to one ASCII space and leading/trailing whitespace
    is removed, matching `whitespace_clean`.
    """
    var source = _str_to_cps(text)
    var cleaned = List[Int]()
    var pending_space = False

    for i in range(len(source)):
        var cp = source[i]
        if is_whitespace(cp) or cp == 0x3000:
            if len(cleaned) > 0:
                pending_space = True
            continue

        if pending_space:
            cleaned.append(0x20)
            pending_space = False

        if cp >= 0xFF01 and cp <= 0xFF5E:
            cp -= 0xFEE0
        cleaned.append(cp)

    return _cps_to_string(cleaned)
