"""把 CPU 数值等价性验证的真实输出渲染成一张终端风格图片（用于 issue 评论）。

只渲染**实际跑出来的数字**（全部从 cpu_equivalence_result.json / cpu_equivalence_out.txt
读入），不写入任何未实测的结论。图片顶部与底部都有醒目的边界声明。
"""

import json
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

BG = (13, 17, 23)
BAR = (22, 27, 34)
FG = (201, 209, 217)
GREEN = (86, 211, 100)
YELLOW = (233, 196, 106)
RED = (248, 81, 73)
BLUE = (121, 192, 255)
GREY = (125, 133, 144)
TITLE = (255, 255, 255)

FONT_SIZE = 17
LINE_H = 25
PAD = 26
BAR_H = 40
WIDTH_CELLS = 98


def is_wide(ch):
    o = ord(ch)
    return (
        0x2E80 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
        or 0xFF00 <= o <= 0xFF60
        or 0xFFE0 <= o <= 0xFFE6
    )


def dwidth(s):
    return sum(2 if is_wide(c) else 1 for c in s)


def wrap(s, width):
    """按显示宽度折行（CJK 宽 2）。优先在空格处断开。"""
    if dwidth(s) <= width:
        return [s]
    out = []
    cur = ""
    cur_w = 0
    for token in re.split(r"(\s+)", s):
        if not token:
            continue
        tw = dwidth(token)
        if cur_w + tw > width and cur.strip():
            out.append(cur.rstrip())
            cur, cur_w = "", 0
            if token.isspace():
                continue
        cur += token
        cur_w += tw
    if cur.strip():
        out.append(cur.rstrip())
    return out


def build_fonts():
    mono = ImageFont.truetype(MONO, FONT_SIZE)
    mono_b = ImageFont.truetype(MONO_B, FONT_SIZE)
    adv = mono.getlength("M")
    cjk = ImageFont.truetype(CJK, int(round(2 * adv)), index=2)
    return mono, mono_b, cjk, adv


def draw_line(draw, x, y, line, mono, mono_b, cjk, adv, color=FG):
    cx = x
    for ch in line:
        if is_wide(ch):
            draw.text((cx, y), ch, font=cjk, fill=color)
            cx += 2 * adv
        else:
            fnt = mono_b if ch.isupper() and ch.isalpha() else mono
            draw.text((cx, y), ch, font=fnt, fill=color)
            cx += adv


def role_of(line):
    s = line.strip()
    if s.startswith("✓"):
        return GREEN
    if s.startswith("×") or s.startswith("✗"):
        return RED
    if s.startswith("⚠"):
        return YELLOW
    if re.match(r"^\[[0-9一-鿿环境]+\]", s):
        return BLUE
    if s.startswith("=") or s.startswith("-"):
        return GREY
    return FG


def render(lines, out_path, title):
    mono, mono_b, cjk, adv = build_fonts()
    width = int(WIDTH_CELLS * adv) + 2 * PAD
    height = BAR_H + PAD + len(lines) * LINE_H + PAD
    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, BAR_H], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([16 + i * 22, 14, 28 + i * 22, 26], fill=c)
    tf = ImageFont.truetype(MONO, 14)
    draw.text(((width - tf.getlength(title)) / 2, 12), title, font=tf, fill=TITLE)

    y = BAR_H + PAD
    for line in lines:
        draw_line(draw, PAD, y, line, mono, mono_b, cjk, adv, role_of(line))
        y += LINE_H
    # 以 JPEG 保存：GitCode 的 raw API 对图片统一返回 image/jpeg，
    # 字节与 Content-Type 一致可避免浏览器端嗅探失败导致图片裂开。
    if out_path.lower().endswith(".jpg") or out_path.lower().endswith(".jpeg"):
        img.convert("RGB").save(out_path, format="JPEG", quality=94, subsampling=0)
    else:
        img.save(out_path)
    return img.size


def boxed(content, width=WIDTH_CELLS):
    """带边框的提示框（内容按显示宽度对齐）。"""
    inner = width - 4
    out = ["┌" + "─" * (width - 2) + "┐"]
    for c in content:
        for piece in wrap(c, inner - 2):
            out.append("│ " + piece + " " * (inner - 1 - dwidth(piece)) + "│")
    out.append("└" + "─" * (width - 2) + "┘")
    return out


def main():
    res = json.load(open(os.path.join(HERE, "cpu_equivalence_result.json")))
    raw = open(os.path.join(HERE, "cpu_equivalence_out.txt")).read().rstrip("\n").split("\n")

    body = raw[3:] if raw and raw[0].startswith("=") else raw
    body = [ln.replace("**", "") for ln in body]
    body = [ln.replace("✗", "×") for ln in body]

    header = boxed([
        "Challenge 09 · MatmulAddRelu —— 自验证报告（本地 CPU 数值等价性验证）",
        "⚠ 本报告不是 Ascend 950 实机 PASS 截图，也不冒充它。",
        "本机无 NPU（无 /dev/davinci0），官方 test_matmul_add_relu.py 未执行。",
        "以下每个数字都是本机 CPU 实算结果；未使用占位图，未引用他人验证图。",
    ]) + [""]

    foot_text = [
        "",
        "复核要点：",
        "1) 复刻的是 PR #4319 中 5 处 TODO 的数据流（K 分 4 片累加 → SPLIT_M 分行 → Add → ReLU → 写回）。",
        "2) 容差公式 max(1,|expected|)/256 与官方验证脚本逐字一致；超差 %d/65536，最大绝对误差 %.3g（容差下界 %.6g）。"
        % (res["mismatches_vs_reference"], res["max_abs_err_vs_reference"], 1.0 / 256.0),
        "3) 反向对照：把 ReLU 误写成「先 ReLU 后 Add」时超差 %d 个元素，说明判据有鉴别力。"
        % res["reverse_control_mismatches"],
        "4) 官方参考解与本 issue 内参赛者公开的 Ascend 950 实机输出，在可见的 %d 个值上逐值一致（%d/%d）。"
        % (res["device_values_checked"], res["device_values_equal"], res["device_values_checked"]),
        "",
        "等价链： 本补全的数据流 ──► 官方参考解 ──► 公开的 Ascend 950 实机输出",
        "",
        "仍缺：Ascend 950 实机 PASS 截图（需人在 HiDevLab 执行 PR 内 verify_hidevlab.sh）。",
        "复现命令： pip install \"torch==2.9.0+cpu\" && python3 verification/cpu_equivalence_check.py",
    ]
    footer = []
    for t in foot_text:
        footer.extend(wrap(t, WIDTH_CELLS - 2))

    lines = header + body + footer
    size = render(lines, os.path.join(HERE, "verification_report.jpg"),
                  "HansonZhang@cann-contest — challenge09 verification (local CPU)")
    print("wrote verification_report.png", size, "lines:", len(lines))


if __name__ == "__main__":
    sys.exit(main())
