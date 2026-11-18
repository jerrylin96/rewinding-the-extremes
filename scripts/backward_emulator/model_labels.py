"""Display names and colours of the two models in the comparison figures.

One place, so no figure can call a model something the others do not.  These
are figure text only: the data keys ("makani", "cbottle"), file names and CLI
flags (--makani-*, --cbottle-*) keep the code names.
"""

MAKANI_LABEL = "Rewind-AR"  # the backward-in-time autoregressive emulator
CBOTTLE_LABEL = "cBottle-video"  # the video-diffusion model it is compared with
MODEL_LABELS = {"makani": MAKANI_LABEL, "cbottle": CBOTTLE_LABEL}
# Line colours, for the same reason: every comparison figure draws a model in
# the same colour.
MODEL_COLORS = {"makani": "#1b6ca8", "cbottle": "#c1440e"}
