# Boogu-Image T2I chat-template tokenizer (Chunk C7b).
#
# Inference-only. Turns a T2I instruction string into the exact `input_ids` the
# Boogu pipeline feeds the Qwen3-VL text encoder.
#
# This reuses the existing pure-Mojo BPE tokenizer `Qwen3Tokenizer` verbatim
# (no BPE reimplementation). The only NEW code here is the chat-template string
# assembly and a thin tokenize driver.
#
# The Boogu oracle (boogu_c7_oracle.py) builds the conditioning with
#   messages = [{role: system, content: SYSTEM_PROMPT_4_T2I},
#               {role: user,   content: instruction}]
#   processor.apply_chat_template(messages, tokenize=True)
# For a text-only T2I request this expands to the Qwen chat template:
#   <|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n
# with NO trailing assistant-generation prompt (the encoder consumes the full
# instruction-conditioning string, not a generation prompt).
#
# The special tokens <|im_start|>(151644) and <|im_end|>(151645) ship in the
# Qwen3-VL tokenizer.json `added_tokens` array with special=true, so
# Qwen3Tokenizer._split_on_specials matches them atomically as single ids.

from serenitymojo.tokenizer.tokenizer import Qwen3Tokenizer


# Fixed system prompt for the Boogu T2I chat template (SYSTEM_PROMPT_4_T2I).
comptime BOOGU_SYSTEM_PROMPT: StaticString = (
    "You are a helpful assistant that generates high-quality images based on"
    " user instructions. The instructions are as follows."
)


# Negative / unconditional system prompt for T2I CFG (SYSTEM_PROMPT_DROP). The
# CFG uncond branch tokenizes this DROP system prompt with an EMPTY user
# instruction (matches boogu_c8_reference.py `encode(..., SYS_DROP, "")`).
comptime BOOGU_SYSTEM_PROMPT_DROP: StaticString = (
    "Describe the key features of the input image (color, shape, size, texture,"
    " objects, background), then explain how the user's text instruction should"
    " alter or modify the image. Generate a new image that meets the user's"
    " requirements while maintaining consistency with the original input where"
    " appropriate."
)


def boogu_chat_template(instruction: String) -> String:
    # Assemble the exact Boogu T2I conditioning string. Specials are written as
    # their literal content (<|im_start|>, <|im_end|>); the tokenizer encodes
    # them atomically because they are registered added/special tokens.
    var s = String("<|im_start|>system\n")
    s += String(BOOGU_SYSTEM_PROMPT)
    s += String("<|im_end|>\n<|im_start|>user\n")
    s += instruction
    s += String("<|im_end|>\n")
    return s^


def boogu_tokenize(tok: Qwen3Tokenizer, instruction: String) raises -> List[Int]:
    # input_ids = encode(chat_template(instruction)). No extra BOS/EOS: the
    # Qwen post-processor adds no special tokens, and the template itself
    # carries every <|im_start|>/<|im_end|>/\n the oracle emits.
    return tok.encode(boogu_chat_template(instruction))


def boogu_chat_template_uncond() -> String:
    # The CFG unconditional conditioning string for T2I: the DROP system prompt
    # with an EMPTY user instruction. Mirrors apply_chat_template on
    #   [{role: system, content: SYSTEM_PROMPT_DROP}, {role: user, content: ""}]
    # which expands to:
    #   <|im_start|>system\n{DROP}<|im_end|>\n<|im_start|>user\n<|im_end|>\n
    # (the empty user body collapses to just the two markers + the \n's).
    var s = String("<|im_start|>system\n")
    s += String(BOOGU_SYSTEM_PROMPT_DROP)
    s += String("<|im_end|>\n<|im_start|>user\n<|im_end|>\n")
    return s^


def boogu_tokenize_uncond(tok: Qwen3Tokenizer) raises -> List[Int]:
    # input_ids for the CFG uncond branch = encode(uncond chat template). Same
    # no-extra-special-tokens contract as boogu_tokenize.
    return tok.encode(boogu_chat_template_uncond())
