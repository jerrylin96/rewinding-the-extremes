"""Display names of the two models in the backward-emulator comparison figures.

One place, so no figure can call a model something the others do not.  These
are figure text only: the data keys ("makani", "cbottle"), file names and CLI
flags (--makani-*, --cbottle-*) keep the code names.
"""

MAKANI_LABEL = "Rewind-AR"  # the backward-in-time autoregressive emulator
CBOTTLE_LABEL = "cBottle-video"  # the video-diffusion model it is compared with
MODEL_LABELS = {"makani": MAKANI_LABEL, "cbottle": CBOTTLE_LABEL}
