"""B1–B3 live inference using the frozen onsite training code and weights."""
from collections import deque
import sys

import cv2
import numpy as np
import torch
from tokenizers import Tokenizer

BEST = "cadfb14f1ee513f42ad7b68169f1a12f23dfcd675273a15387c572eb3b840d22"


class Predictor:
    def __init__(self, work, task):
        run = work / "offline_training/b123_goal_v1"
        sys.path.insert(0, str(run))
        from piper_base import EasyDict, Frontend, GOAL_SHA, HISTORY, Policy, TOKEN_SHA, digest
        paths = [run/name for name in ("best.pt", "goal_multitask_model_ep96.pth", "clip_tokenizer_goal.json")]
        if [digest(p) for p in paths] != [BEST, GOAL_SHA, TOKEN_SHA]:
            raise ValueError("模型或分词表校验失败")
        saved = torch.load(paths[0], map_location="cpu", weights_only=True)
        config = saved["config"]
        if digest(run/"piper_base.py") != config["script_sha256"]:
            raise ValueError("训练代码版本不匹配")
        with torch.serialization.safe_globals([(EasyDict, "easydict.EasyDict")]):
            ckpt = torch.load(paths[1], map_location="cpu", weights_only=True)
        state = {k.removeprefix("module."): v for k, v in ckpt["state_dict"].items()}
        self.front = Frontend(state, config["lora_scale"], Tokenizer.from_file(str(paths[2]))).cuda().eval()
        self.model = Policy(state, self.front.language.cpu(), config["normalization"])
        self.model.load_state_dict(saved["policy"], strict=True)
        self.model.cuda().eval()
        self.task = task
        self.tag = torch.tensor([task], device="cuda")
        self.history = deque(maxlen=HISTORY)

    @torch.inference_mode()
    def predict(self, images, feedback):
        if len(images) != 2 or any(im.shape != (480,640,3) or im.dtype != np.uint8 for im in images):
            raise ValueError("需要 640×480 的俯视和腕部 BGR 图像")
        if len(feedback) != 7 or not np.isfinite(feedback).all():
            raise ValueError("反馈无效")
        pixels = []
        for image in images:
            small = cv2.resize(image, (224,168), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pixels.append(cv2.copyMakeBorder(rgb,28,28,0,0,cv2.BORDER_CONSTANT,value=(123,117,104)))
        x = torch.from_numpy(np.stack(pixels)).cuda().permute(0,3,1,2).float()/255
        f = self.front
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = f.vision(pixel_values=(x-f.mean)/f.std).pooler_output
            lang = f.language[self.task]
            v = (f.proj(v)*(1+f.scale(lang))+f.shift(lang)).half()
            self.history.append((v, torch.as_tensor(feedback, dtype=torch.float32, device="cuda")))
            seq = [self.history[0]]*(self.history.maxlen-len(self.history)) + list(self.history)
            vision, q = (torch.stack(s)[None] for s in zip(*seq))
            _, target = self.model(vision, q, self.tag)
        result = target[0].tolist()
        if not np.isfinite(result).all():
            raise ValueError("策略输出非有限数值")
        return result
