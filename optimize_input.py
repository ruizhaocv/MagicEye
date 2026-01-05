import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import datasets
from networks import define_G 

# -----------------------
# Utils
# -----------------------
def save_vis(out_dir, step, x_img, pred, gt):
    """
    x_img: [1,3,H,W] in [0,1]
    pred/gt: [1,1,H,W] (range depends on your dataset)
    """
    os.makedirs(out_dir, exist_ok=True)

    x = x_img.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()  # HWC
    p = pred.detach().cpu()[0, 0].numpy()
    g = gt.detach().cpu()[0, 0].numpy()

    # normalize pred/gt for visualization only
    def norm01(a):
        a = a.astype(np.float32)
        mn, mx = np.min(a), np.max(a)
        if mx - mn < 1e-8:
            return np.zeros_like(a)
        return (a - mn) / (mx - mn)

    p_vis = norm01(p)
    g_vis = norm01(g)

    # concatenate: input / pred / gt vertically
    fig = plt.figure(figsize=(6, 9))
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.imshow(x)
    ax1.set_title("Learnable input (clamped to [0,1])")
    ax1.axis("off")

    ax2 = fig.add_subplot(3, 1, 2)
    ax2.imshow(p_vis, cmap="gray")
    ax2.set_title("Pred depth (normalized for vis)")
    ax2.axis("off")

    ax3 = fig.add_subplot(3, 1, 3)
    ax3.imshow(g_vis, cmap="gray")
    ax3.set_title("GT depth (normalized for vis)")
    ax3.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"step_{step:06d}.png"), dpi=150)
    plt.close(fig)


def load_checkpoint(net_G, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    # 你的 ckpt 里 key 是 model_G_state_dict
    net_G.load_state_dict(ckpt["model_G_state_dict"], strict=True)
    return ckpt


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser()

    # must match your training config
    parser.add_argument("--dataset", type=str, default="mnist",
                        help="same as training: mnist/shapenet/watermarking/...")
    parser.add_argument("--net_G", type=str, default="vit_b_16_fcn",
                        help="same as training, e.g., vit_b_16_fcn / unet_256 / resnet18fcn")
    parser.add_argument("--norm_type", type=str, default="batch")
    parser.add_argument("--with_disparity_conv", action="store_true", default=False)
    parser.add_argument("--with_skip_connection", action="store_true", default=False)  # for resnet fcn
    parser.add_argument("--in_size", type=int, default=256)

    # checkpoint
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_ckpt.pt",
                        help="path to best_ckpt.pt or last_ckpt.pt")

    # input optimization hyperparams
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr_img", type=float, default=5e-2)
    parser.add_argument("--lambda_tv", type=float, default=0.0,
                        help="optional TV regularization weight (0 disables)")
    parser.add_argument("--vis_every", type=int, default=100)
    parser.add_argument("--out_dir", type=str, default="./input_opt_out")

    # whether to start from the real stereogram (gt input) or from noise
    parser.add_argument("--init", type=str, default="gt", choices=["gt", "noise"])

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1) build dataloaders and get the first sample as GT
    dataloaders = datasets.get_loaders(args)

    # 从 val 的第一个 batch 取第一个 sample
    first_batch = next(iter(dataloaders["val"]))
    gt_stereo = first_batch["stereogram"][0:1].to(device)  # [1,3,H,W]
    gt_depth  = first_batch["dmap"][0:1].to(device)        # [1,1,H,W]

    # 2) build net and load ckpt
    net_G = define_G(args).to(device)
    ckpt = load_checkpoint(net_G, args.checkpoint_path, device)

    net_G.eval()
    for p in net_G.parameters():
        p.requires_grad_(False)

    # 3) create learnable input
    if args.init == "gt":
        x_param = gt_stereo.clone().detach()
    else:
        x_param = torch.rand_like(gt_stereo)

    # 关键：learnable tensor
    x_param = torch.nn.Parameter(x_param)

    # 4) optimize input to match GT depth
    loss_fn = nn.MSELoss()
    opt = optim.Adam([x_param], lr=args.lr_img)

    # optional: TV regularization for smoother image
    def tv_loss(x):
        # x: [B,C,H,W]
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
        return dh + dw

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "meta.txt"), "w") as f:
        f.write(f"checkpoint: {args.checkpoint_path}\n")
        f.write(f"epoch_id: {ckpt.get('epoch_id', 'NA')}\n")
        f.write(f"net_G: {args.net_G}\n")
        f.write(f"init: {args.init}\n")
        f.write(f"steps: {args.steps}, lr_img: {args.lr_img}, lambda_tv: {args.lambda_tv}\n")

    # also save the GT depth visualization once
    save_vis(args.out_dir, step=0, x_img=gt_stereo, pred=net_G(gt_stereo), gt=gt_depth)

    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)

        # clamp only for forward (保持可训练同时避免值爆掉)
        x_in = x_param.clamp(0.0, 1.0)

        pred = net_G(x_in)
        loss = loss_fn(pred, gt_depth)

        if args.lambda_tv > 0:
            loss = loss + args.lambda_tv * tv_loss(x_in)

        loss.backward()
        opt.step()

        if step % 10 == 0 or step == 1:
            print(f"[step {step:05d}/{args.steps}] loss={loss.item():.6f}")

        if step % args.vis_every == 0 or step == args.steps:
            with torch.no_grad():
                pred_vis = net_G(x_param.clamp(0.0, 1.0))
            save_vis(args.out_dir, step=step, x_img=x_param.clamp(0, 1), pred=pred_vis, gt=gt_depth)

    # save final learned input tensor
    torch.save(
        {
            "x_learned": x_param.detach().cpu(),
            "gt_depth": gt_depth.detach().cpu(),
            "gt_stereogram": gt_stereo.detach().cpu(),
            "args": vars(args),
        },
        os.path.join(args.out_dir, "final.pt"),
    )

    print(f"Done. Outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
