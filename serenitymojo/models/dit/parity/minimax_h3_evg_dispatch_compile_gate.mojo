# Compile/link surface for the opt-in EVG H3 attention dispatcher.  This gate
# imports the real DiT module without instantiating the full product pipeline,
# keeping compiler memory bounded while training owns the machine.

from serenitymojo.models.dit.minimax_h3_dit import MINIMAX_H3_ATTN_EVG_INT8


def main():
    print("MiniMax-H3 EVG attention backend=", MINIMAX_H3_ATTN_EVG_INT8)
