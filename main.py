import argparse
import copy
import pdb

import torch
import torch.nn as nn
from torchvision.utils import save_image

from ultralytics import YOLO

def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    return total

def save_as_ultralytics_pt(yolo_obj, src_pt: str, dst_pt: str):
    ckpt = torch.load(src_pt, map_location="cpu", weights_only=False)

    net = getattr(yolo_obj, "model", yolo_obj)
    if hasattr(net, "module"):
        net = net.module

    net_to_save = copy.deepcopy(net).cpu()
    ckpt["model"] = net_to_save

    torch.save(ckpt, dst_pt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .pt model")
    parser.add_argument("--target", required=True, help="image path or glob or dir")
    args = parser.parse_args()

    model = YOLO(args.model)    
    results = model([args.target])

    # save_as_ultralytics_pt(model, 'models/snndet.pt', 'models/snndet_0.8B.pt')

    total = count_params(model)
    print("Total params:", total, "|", f"{total / 1_000_000_000:.3f} B")

    for i, r in enumerate(results):
        im_bgr = r.plot()
        im_rgb = im_bgr[..., ::-1].copy()
        img_t = torch.from_numpy(im_rgb).permute(2, 0, 1).float().div(255.0)
        save_image(img_t, f"result_{i:04d}.jpg")  # result_0000.jpg ...

if __name__ == "__main__":
    main()
