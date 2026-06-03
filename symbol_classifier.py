"""
symbol_classifier.py
CNN architecture, class definitions, and label-mapping utilities
for musical symbol classification.

the vocabulary constants (CLASSES, CLASS_TO_IDX, etc.) and
semantic_token_to_class() are importable without torch.
SymbolCNN is built lazily so torch does not need to be installed
unless you are actually training or running inference.
"""


# ---------------------------------------------------------------------------
# class vocabulary
# scoped to this project: monophonic scores with whole/half/quarter/eighth
# notes, their rest equivalents, and an "other" bucket for clefs, barlines,
# time signatures, accidentals, ties, etc.
# ---------------------------------------------------------------------------
CLASSES = [
    "note_whole",
    "note_half",
    "note_quarter",
    "note_eighth",
    "rest_whole",
    "rest_half",
    "rest_quarter",
    "rest_eighth",
    "other",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
N_CLASSES    = len(CLASSES)

# every crop is resized to this square before entering the network
IMG_SIZE = 64


def semantic_token_to_class(token: str) -> str:
    """
    map a single PrIMuS semantic token to one of the CLASSES strings.

    PrIMuS semantic tokens for notes follow the pattern:
        note-<pitch><octave>_<duration>[.<modifier>]
    e.g. note-C4_quarter, note-G5_half., note-E3_eighth

    rest tokens:  rest-whole, rest-half, rest-quarter, rest-eighth

    everything else (clef-, keySignature-, timeSig-, barline, etc.) -> "other"
    """
    t = token.strip()
    if t.startswith("note-"):
        for dur in ("whole", "half", "quarter", "eighth"):
            if f"_{dur}" in t:
                return f"note_{dur}"
        return "other"
    if t.startswith("rest-"):
        for dur in ("whole", "half", "quarter", "eighth"):
            if dur in t:
                return f"rest_{dur}"
        return "other"
    return "other"


# ---------------------------------------------------------------------------
# CNN architecture  (torch imported lazily so this module loads without it)
# ---------------------------------------------------------------------------

def _build_cnn_classes():
    """
    build and return (_ConvBlock, SymbolCNN) using the locally imported torch.
    called once on first use of SymbolCNN.
    """
    import torch
    import torch.nn as nn

    class _ConvBlock(nn.Module):
        """conv -> batch norm -> relu -> optional max-pool."""
        def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
            super().__init__()
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class SymbolCNN(nn.Module):
        """
        4-layer convolutional classifier for 1-channel IMG_SIZE x IMG_SIZE crops.

        spatial flow (for IMG_SIZE=64):
            1 x 64 x 64
            -> block1 (pool): 32 x 32 x 32
            -> block2 (pool): 64 x 16 x 16
            -> block3 (pool): 128 x  8 x  8
            -> block4 (pool): 256 x  4 x  4
            -> flatten: 4096
            -> fc1 (512) -> relu -> dropout(0.5)
            -> fc2 (n_classes)

        batch norm after every conv prevents gradient issues during training on
        small crops; dropout on the first fc layer is the main regularizer.
        """

        def __init__(self, n_classes: int = N_CLASSES, img_size: int = IMG_SIZE):
            super().__init__()
            self.features = nn.Sequential(
                _ConvBlock(1,   32, pool=True),
                _ConvBlock(32,  64, pool=True),
                _ConvBlock(64, 128, pool=True),
                _ConvBlock(128, 256, pool=True),
            )
            feature_dim = 256 * (img_size // 16) ** 2
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feature_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, n_classes),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return self.classifier(self.features(x))

    return SymbolCNN


# expose SymbolCNN at module level; instantiation triggers the lazy import
class SymbolCNN:
    """
    proxy that builds the real nn.Module class on first instantiation.
    import this class freely — torch is only required when you call SymbolCNN().
    """
    _real_class = None

    def __new__(cls, *args, **kwargs):
        if cls._real_class is None:
            cls._real_class = _build_cnn_classes()
        return cls._real_class(*args, **kwargs)
