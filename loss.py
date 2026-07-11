"""
RCLane loss -- ported to PyTorch from src/rclane/rclane_loss.py (MindSpore).

Components:
  - seg: cross-entropy with OHEM (hard-negative mining), neg:pos ratio = 15:1
  - up/down arrow, up/down bound: SmoothL1, computed ONLY at foreground pixels
  - every term is normalized by the number of positive pixels, summed with equal weight

Layout convention (matches rclane.py output, NCHW):
  preds[k]:  (B, 2, H, W)
      - seg_map: 2-class logits [background, lane]
      - arrow/bound: 2 regression channels
  targets[k]:
      - seg_map: (B, H, W) with values {0,1}, 1 = lane   (or one-hot (B,2,H,W))
      - arrow/bound: (B, 2, H, W) GT values (from encode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RCLaneLoss(nn.Module):
    def __init__(self, negative_ratio: int = 15, alpha: float = 1.0,
                 smooth_l1_beta: float = 1.0):
        super().__init__()
        self.negative_ratio = negative_ratio
        self.alpha = alpha
        self.beta = smooth_l1_beta

    @staticmethod
    def _pos_mask(gt_seg):
        """Accept GT seg as (B,H,W) {0,1} or one-hot (B,2,H,W)."""
        if gt_seg.dim() == 4 and gt_seg.shape[1] == 2:
            gt_seg = gt_seg[:, 1]  # lane channel
        return gt_seg > 0.5  # (B,H,W) bool

    def _reg_loss(self, pred, gt, pmask, pos_num):
        """SmoothL1 at foreground pixels only. pred/gt: (B,2,H,W)."""
        pred_pos = pred.permute(0, 2, 3, 1)[pmask]  # (Npos, 2)
        gt_pos = gt.permute(0, 2, 3, 1)[pmask]
        if pred_pos.numel() == 0:
            return pred.sum() * 0.0  # keep grad graph, value 0
        loss = F.smooth_l1_loss(pred_pos, gt_pos, beta=self.beta, reduction="sum")
        return loss / pos_num

    def forward(self, preds: dict, targets: dict) -> dict:
        pr_seg = preds["seg_map"]  # (B,2,H,W) logits
        pmask = self._pos_mask(targets["seg_map"])  # (B,H,W)
        nmask = ~pmask

        fpmask = pmask.float()
        fnmask = nmask.float()
        pos_total = fpmask.sum()
        neg_total = fnmask.sum()

        pos_num = pos_total.clamp(min=1)
        neg_num = int((pos_total * self.negative_ratio).clamp(min=1, max=neg_total).item())

        # log-softmax over the channel dim
        logp = F.log_softmax(pr_seg, dim=1)  # (B,2,H,W)
        bg_logp = logp[:, 0]  # log P(background)  (B,H,W)
        fg_logp = logp[:, 1]  # log P(lane)

        # --- positive CE ---
        loss_pos = -(self.alpha * fg_logp * fpmask).sum() / pos_num

        # --- negative CE with OHEM: keep the neg_num "hardest" negative pixels ---
        # hard = model thinks it is lane => lowest bg_logp among negative pixels
        neg_logp = bg_logp[nmask]  # (Nneg,)
        if neg_logp.numel() > 0 and neg_num > 0:
            k = min(neg_num, neg_logp.numel())
            # the neg_num SMALLEST bg_logp  <=>  the LARGEST (-bg_logp)
            hard_vals, _ = torch.topk(-neg_logp, k)
            loss_neg = (self.alpha * hard_vals).sum() / pos_num
        else:
            loss_neg = bg_logp.sum() * 0.0

        # --- arrow / bound regression ---
        up_arrow_loss = self._reg_loss(preds["up_arrow"], targets["up_arrow"], pmask, pos_num)
        down_arrow_loss = self._reg_loss(preds["down_arrow"], targets["down_arrow"], pmask, pos_num)
        up_bound_loss = self._reg_loss(preds["up_bound"], targets["up_bound"], pmask, pos_num)
        down_bound_loss = self._reg_loss(preds["down_bound"], targets["down_bound"], pmask, pos_num)

        total = (loss_pos + loss_neg + up_arrow_loss + down_arrow_loss
                 + up_bound_loss + down_bound_loss)

        return {
            "loss": total,
            "seg_pos": loss_pos,
            "seg_neg": loss_neg,
            "up_arrow": up_arrow_loss,
            "down_arrow": down_arrow_loss,
            "up_bound": up_bound_loss,
            "down_bound": down_bound_loss,
        }


# --------------------------------------------------------------------------- #
#  smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from rclane import RCLane

    torch.manual_seed(0)
    B, H, W = 2, 320, 800
    model = RCLane(vision="b0", img_size=(H, W))
    crit = RCLaneLoss()

    x = torch.randn(B, 3, H, W)
    preds = model(x)

    # synthetic GT: random ~5% foreground seg, random arrow/bound
    gt_seg = (torch.rand(B, H, W) < 0.05).long()
    targets = {
        "seg_map": gt_seg,
        "up_arrow": torch.randn(B, 2, H, W),
        "down_arrow": torch.randn(B, 2, H, W),
        "up_bound": torch.rand(B, 2, H, W) * 3,
        "down_bound": torch.rand(B, 2, H, W) * 3,
    }

    out = crit(preds, targets)
    for k, v in out.items():
        print(f"  {k:12s} {v.item():.4f}")

    assert torch.isfinite(out["loss"]), "loss is not finite!"
    out["loss"].backward()  # check gradient flow
    g = model.head.out_seg[-1].conv.weight.grad
    assert g is not None and torch.isfinite(g).all(), "broken gradient!"

    # edge case: an image with NO lanes
    targets0 = dict(targets)
    targets0["seg_map"] = torch.zeros(B, H, W).long()
    out0 = crit(model(x), targets0)
    assert torch.isfinite(out0["loss"]), "loss is NaN when there is no foreground!"
    print(f"  [no-lane] loss = {out0['loss'].item():.4f}")

    print("OK -- loss is finite, backward runs, no-lane image handled.")
