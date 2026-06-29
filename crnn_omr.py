"""
crnn_omr.py
end-to-end CRNN for optical music recognition of a single monophonic staff.

architecture (Calvo-Zaragoza & Rizo, 2018, the model PrIMuS was built for):

    staff image  ->  CNN encoder  ->  collapse height  ->  width-indexed
    feature sequence  ->  BiLSTM  ->  per-timestep logits over (vocab + blank)
    ->  CTC loss / greedy decode  ->  semantic token sequence

why this and not segment-then-classify:
the old pipeline cuts a staff into connected-component crops and labels each one.
every segmentation merge (beamed notes) or fragment (severed stem) permanently
destroys a note before the classifier ever sees it.  a CRNN reads the *whole*
staff left-to-right and is trained with CTC, which is alignment-free: it never
needs to know where one symbol ends and the next begins, so there is no
segmentation stage to cascade errors.

input convention
----------------
- grayscale, ink-dark-on-light, fixed height H (default 96), variable width W.
- a tensor of shape (B, 1, H, W); widths may differ, so batches are right-padded
  and the true (unpadded) width of each sample drives the CTC input lengths.

the width is downsampled by a factor of 4 through the encoder (two width-pooling
steps), so a staff of width W yields T = W // 4 time-steps.  keep the width
pooling gentle (4x, not 16x) so T stays >= the token-sequence length, which CTC
requires.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# width is pooled at the first two blocks only (2x * 2x = 4x total)
WIDTH_DOWNSAMPLE = 4


class CRNN(nn.Module):
    def __init__(self, n_classes: int, img_height: int = 96, lstm_hidden: int = 256,
                 lstm_layers: int = 2, dropout: float = 0.2):
        """
        n_classes : size of the token vocabulary *excluding* the CTC blank.
                    the output layer emits n_classes + 1 logits; index
                    `n_classes` (the last one) is reserved as the blank.
        img_height: fixed input height H.  must be divisible by 16 (four 2x
                    height-pools) so the encoder collapses height cleanly.
        """
        super().__init__()
        if img_height % 16 != 0:
            raise ValueError(f"img_height must be divisible by 16, got {img_height}")
        self.img_height = img_height
        self.n_classes = n_classes
        self.blank = n_classes  # CTC blank id = last index

        def block(cin, cout, pool):
            # conv -> batchnorm -> relu -> maxpool(pool)
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        # height is pooled 2x in every block (96 -> 48 -> 24 -> 12 -> 6).
        # width is pooled 2x only in the first two blocks (total 4x), then 1x,
        # so we keep enough horizontal time-steps for CTC.
        self.cnn = nn.Sequential(
            block(1,   32, (2, 2)),   # H/2,  W/2
            block(32,  64, (2, 2)),   # H/4,  W/4
            block(64, 128, (2, 1)),   # H/8,  W/4
            block(128, 256, (2, 1)),  # H/16, W/4
        )
        # after four 2x height pools, H=96 -> 6.  collapse those 6 rows to 1
        # with an adaptive pool so the feature map becomes a pure 1-D sequence.
        self.collapse = nn.AdaptiveAvgPool2d((1, None))

        self.rnn = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
            bidirectional=True, batch_first=True, dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(lstm_hidden * 2, n_classes + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, H, W)
        returns log-probabilities of shape (T, B, n_classes + 1) — the layout
        torch.nn.CTCLoss expects (time-major).
        """
        feat = self.cnn(x)              # (B, 256, H/16, W/4)
        feat = self.collapse(feat)      # (B, 256, 1, W/4)
        feat = feat.squeeze(2)          # (B, 256, T)
        feat = feat.permute(0, 2, 1)    # (B, T, 256)
        seq, _ = self.rnn(feat)         # (B, T, 2*hidden)
        logits = self.fc(seq)           # (B, T, n_classes + 1)
        logp = logits.log_softmax(dim=2)
        return logp.permute(1, 0, 2)    # (T, B, n_classes + 1)

    def time_steps(self, width: int) -> int:
        """number of output time-steps T for an input of the given pixel width."""
        return width // WIDTH_DOWNSAMPLE


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick shape sanity check
    m = CRNN(n_classes=200, img_height=96)
    dummy = torch.zeros(2, 1, 96, 512)
    out = m(dummy)
    print("params:", count_params(m))
    print("input (2,1,96,512) -> output", tuple(out.shape),
          "(expect T=128, B=2, C=201)")
