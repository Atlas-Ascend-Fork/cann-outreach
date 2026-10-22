#!/usr/bin/env python3
"""Generate the installed CANN version's project, overlay sources, and build it."""

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soc", default="Ascend910B1", help="例如 Ascend910B1 / Ascend910B4")
    parser.add_argument("--generate-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"Ascend910B[1-4]?", args.soc):
        parser.error("当前实现只注册了 Ascend 910B 系列")

    toolkit_env = os.environ.get("ASCEND_CANN_PACKAGE_PATH") or os.environ.get("ASCEND_HOME_PATH")
    toolkit = Path(toolkit_env).resolve() if toolkit_env else None
    msopgen = shutil.which("msopgen")
    if msopgen is None and toolkit:
        candidate = toolkit / "python/site-packages/bin/msopgen"
        if candidate.is_file():
            msopgen = str(candidate)
    if msopgen is None:
        parser.error("找不到 msopgen，请先 source <ascend-toolkit>/set_env.sh")

    project = ROOT / "build/ReluCustom"
    marker = project / ".relu_custom_generated"
    if project.exists() and not marker.is_file():
        parser.error(f"{project} 已存在且不是本脚本生成的工程，请移动该目录后重试")
    if not project.exists():
        project.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([msopgen, "gen", "-i", str(ROOT / "relu_custom.json"),
                        "-c", f"ai_core-{args.soc}", "-lan", "cpp", "-out", str(project)],
                       check=True)
        marker.write_text("ReluCustom generated project\n", encoding="utf-8")

    for directory in ("op_host", "op_kernel"):
        for source in (ROOT / directory).iterdir():
            if source.is_file():
                shutil.copy2(source, project / directory / source.name)

    presets_path = project / "CMakePresets.json"
    presets = json.loads(presets_path.read_text(encoding="utf-8"))
    cache = presets["configurePresets"][0]["cacheVariables"]
    if toolkit:
        cache["ASCEND_CANN_PACKAGE_PATH"] = {"type": "PATH", "value": str(toolkit)}
    cache["ASCEND_COMPUTE_UNIT"] = {"type": "STRING", "value": "ascend910b"}
    cache["ENABLE_SOURCE_PACKAGE"] = {"type": "BOOL", "value": "True"}
    cache["ENABLE_BINARY_PACKAGE"] = {"type": "BOOL", "value": "True"}
    cache["ENABLE_TEST"] = {"type": "BOOL", "value": "False"}
    presets_path.write_text(json.dumps(presets, indent=4) + "\n", encoding="utf-8")
    print(f"工程已生成：{project}", flush=True)

    if not args.generate_only:
        missing = [name for name in ("cmake", "make", "g++") if shutil.which(name) is None]
        if missing:
            parser.error("工程已生成，但缺少构建依赖：" + ", ".join(missing))
        subprocess.run(["bash", "build.sh"], cwd=project, check=True)
        print(f"算子包输出目录：{project / 'build_out'}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
