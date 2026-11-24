"""Review paired archived scene and model images. All output images stay local."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RUNS = [
    ("20261123_234228_1446439_B1-ode50-20261123", "23:48:25", (248, 350), (0.02, 1.10, 2.73)),
    ("20261123_234953_1448999_B1-ode50-20261123", "23:50:10", (218, 357), (0.02, 2.52, 4.45)),
]


def inspect(folder, clock, center, times, sheet, row):
    index = json.loads((folder / "index.json").read_text(encoding="utf-8"))
    active = [item for item in index if item["t"] >= 0]
    selected = {min(active, key=lambda r: abs(r["t"] - t))["i"] for t in times}
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    x, y = center
    mx, my = x // 5, (y + 80) // 5
    previous_stamp = -1
    for item in active:
        if item["stamp"] <= previous_stamp:
            raise ValueError("Non-increasing input timestamp")
        previous_stamp = item["stamp"]
        prefix = f"{item['i']:03d}"
        with Image.open(folder / f"{prefix}_scene.jpg") as scene, Image.open(folder / f"{prefix}_model.png") as model:
            scene.load()
            model.load()
            if (scene.size, model.size) != ((1280, 480), (256, 128)):
                raise ValueError("Unexpected paired image dimensions")
            if item["i"] in selected:
                top = 48 + row * 153
                draw.text((9, top + 50), clock, font=font, fill="black")
                draw.text((9, top + 76), f"+{item['t']:.2f} s", font=font, fill="black")
                crop = scene.crop((x-80, y-60, x+80, y+60))
                small = model.crop((mx-16, my-12, mx+16, my+12))
                wrist = scene.crop((640, 0, 1280, 480))
                sheet.paste(crop.resize((192, 144), Image.Resampling.BILINEAR), (115, top))
                sheet.paste(small.resize((192, 144), Image.Resampling.NEAREST), (323, top))
                sheet.paste(wrist.resize((192, 144), Image.Resampling.LANCZOS), (531, top))
                row += 1
    return {"execution_start": clock, "paired_inputs": len(active),
            "first_input_s": round(active[0]["t"], 3), "last_input_s": round(active[-1]["t"], 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="Private offline extraction with indexed original images")
    parser.add_argument("local_sheet", type=Path, help="Private output image path")
    args = parser.parse_args()
    sheet = Image.new("RGB", (739, 966), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for x, label in [(9, "Start / elapsed"), (115, "Overhead: original crop"),
                     (323, "Same crop: model pixels"), (531, "Wrist: full frame")]:
        draw.text((x, 15), label, fill="black", font=font)
    result = []
    row = 0
    for name, clock, center, times in RUNS:
        result.append(inspect(args.snapshot / name, clock, center, times, sheet, row))
        row += len(times)
    sheet.save(args.local_sheet)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
