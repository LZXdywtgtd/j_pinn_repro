"""
jpinn.py —— J-PINN 统一 CLI 入口（v0.8 阶段 0 Stage 3）

子命令（参数透传给原 CLI）：
  train        训练 PINN 模型
  visualize    可视化预测 vs 真值
  j_integral   后处理 J 积分
  generate_data  生成合成数据
  ablations    跑消融实验编排
  ensemble     多 restart 集成
  pareto       收集 Pareto 数据

用法：
  python jpinn.py train --epochs 5000 --ablation full
  python jpinn.py visualize --checkpoint checkpoints/jpinn.pt
  python jpinn.py j_integral --checkpoint checkpoints/jpinn.pt --anchor_mode residual_min

兼容：旧 CLI 入口（train.py / visualize.py / run_*.py / collect_pareto.py）仍可独立运行。
"""
import argparse
import sys
from pathlib import Path

# 让 jpinn_core/ + data/ + models/ + postprocess/ 可被发现
_CURRENT_DIR = Path(__file__).resolve().parent
for sub in ("jpinn_core", "data", "models", "postprocess"):
    p = str(_CURRENT_DIR / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


# 命令 → (可调用包, 调用函数, 替代入口名)
ROUTES = {
    "train":         ("train",                    "main", "train.py"),
    "visualize":     ("visualize",                "main", "visualize.py"),
    "j_integral":    ("postprocess.run_j_integral", "main", "postprocess.run_j_integral"),
    "generate_data": ("data.generate_synthetic_thermal_data", "main", "data.generate_synthetic_thermal_data"),
    "ablations":     ("run_ablations",            "main", "run_ablations.py"),
    "ensemble":      ("run_ensemble",             "main", "run_ensemble.py"),
    "pareto":        ("collect_pareto",           "main", "collect_pareto.py"),
}


def _dispatch(cmd: str, remaining_args: list) -> int:
    """路由到子命令实现；参数透传关键：把 remaining 塞回 sys.argv 让目标的 argparse 接管"""
    if cmd not in ROUTES:
        print(f"[ERROR] 未知子命令: {cmd}\n可用: {', '.join(ROUTES.keys())}", file=sys.stderr)
        return 2
    module_path, func_name, script_name = ROUTES[cmd]
    # 导入目标模块
    module = __import__(module_path, fromlist=[func_name])
    func = getattr(module, func_name)
    # ★ 关键：把 jpinn.py 截留的子命令参数塞回 sys.argv
    # 目标脚本的 argparse 看到 sys.argv[0] = script_name + 子命令参数 → 正常解析
    sys.argv = [script_name] + remaining_args
    # 调用原 main()
    result = func()
    return int(result) if isinstance(result, int) else 0


def main() -> int:
    # 简化策略：jpinn 不解析子命令的 -h/--help，直接用 parser.parse_known_args
    # 把第一个位置参数当子命令，其余透传
    if len(sys.argv) < 2:
        print("用法: python jpinn.py <command> [args...]\n", file=sys.stderr)
        print("可用命令: " + ", ".join(ROUTES.keys()), file=sys.stderr)
        return 2
    cmd = sys.argv[1]
    if cmd in ("-h", "--help"):
        # 显示 jpinn 自身帮助
        parser = argparse.ArgumentParser(prog="jpinn", description="J-PINN 统一 CLI 入口（旧 CLI 仍兼容）")
        subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
        for c in ROUTES:
            subparsers.add_parser(c, help=f"运行 {c}（参数透传）")
        parser.print_help()
        return 0
    if cmd not in ROUTES:
        print(f"[ERROR] 未知子命令: {cmd}\n可用: {', '.join(ROUTES.keys())}\n", file=sys.stderr)
        print("提示: 命令的 --help 直接透传，如: python jpinn.py train --help", file=sys.stderr)
        return 2
    # 透传：remaining = sys.argv[2:]（子命令的所有参数）
    return _dispatch(cmd, sys.argv[2:])


if __name__ == "__main__":
    sys.exit(main())
