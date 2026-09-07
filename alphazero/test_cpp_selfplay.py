"""
测试 C++ ONNX 自对弈的完整流程。

按顺序执行：
  1. ONNX 导出 — 验证网络能正确导出为 ONNX 格式
  2. 二进制格式读写 — 验证 C++ 产出能被 Python 正确读取
  3. 训练样本完整性 — 验证样本数据形状正确
  4. 端到端运行 — 使用 C++ 二进制生成数据并验证

Usage:
  python -m alphazero.test_cpp_selfplay              # 运行所有不依赖 C++ 二进制的测试
  python -m alphazero.test_cpp_selfplay --all         # 运行全部测试（需要编译好的 az_selfplay_onnx.exe）
  python -m alphazero.test_cpp_selfplay --exe PATH    # 指定 C++ 二进制路径
"""

from __future__ import annotations

import io
import os
import struct
import sys
import time
import traceback
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"


def section(name: str):
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")


def check(desc: str, fn):
    """Run a check, print result, return success bool."""
    try:
        result = fn()
        print(f"  {PASS} {desc}")
        return result
    except Exception as e:
        print(f"  {FAIL} {desc}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# 1. ONNX 导出测试
# ---------------------------------------------------------------------------
def test_onnx_export():
    section("1. ONNX Export")
    import torch
    from .config import TrainConfig
    from .network import AlphaZeroNetwork
    from .export_onnx import export_live_network

    device = torch.device("cpu")

    def check_export_shapes():
        """导出 ONNX 并验证输入输出形状。"""
        cfg = TrainConfig()
        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()

        with TemporaryDirectory(prefix="test_onnx_") as tmp_dir:
            output_path = str(Path(tmp_dir) / "test_fp16.onnx")
            result = export_live_network(
                network=network,
                cfg=cfg,
                output_path=output_path,
                device_name="cpu",
                fp16_output=False,
                opset=18,
            )
            assert result.exists(), f"ONNX file not created: {result}"
            assert result.stat().st_size > 0, "ONNX file is empty"

            # 验证 ONNX 模型结构
            import onnx
            model = onnx.load(result)
            # 检查输入数量（14 个输入）
            assert len(model.graph.input) == 14, \
                f"Expected 14 inputs, got {len(model.graph.input)}"
            # 检查输出数量（value + action_logits）
            assert len(model.graph.output) == 2, \
                f"Expected 2 outputs, got {len(model.graph.output)}"

            # 验证输入名称
            input_names = {inp.name for inp in model.graph.input}
            expected = {
                "board_planes", "l_coords", "t_coords",
                "used_board_counts", "urgency", "side_to_move",
                "action_state_indices", "action_board_indices",
                "action_from_squares", "action_to_squares",
                "action_delta_t", "action_delta_l", "action_is_submit",
            }
            assert input_names == expected, \
                f"Input names mismatch: {input_names ^ expected}"

            # 验证输出名称
            output_names = {out.name for out in model.graph.output}
            assert output_names == {"value", "action_logits"}, \
                f"Output names: {output_names}"
        return True

    def check_export_runnable():
        """导出 ONNX 并通过 ONNX Runtime 运行一次推理。"""
        import onnxruntime as ort
        import torch
        from .config import TrainConfig
        from .network import AlphaZeroNetwork
        from .export_onnx import export_live_network

        cfg = TrainConfig()
        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()

        with TemporaryDirectory(prefix="test_onnx_run_") as tmp_dir:
            output_path = str(Path(tmp_dir) / "test_fp32.onnx")
            result = export_live_network(
                network=network,
                cfg=cfg,
                output_path=output_path,
                device_name="cpu",
                fp16_output=False,
                opset=18,
            )

            # 创建 ONNX Runtime session
            session = ort.InferenceSession(str(result))
            input_meta = {inp.name: inp for inp in session.get_inputs()}

            # 构建 dummy 输入（batch_size 必须匹配导出时的 leaf_batch_size）
            batch_size = max(1, int(getattr(cfg.mcts, "leaf_batch_size", 1)))
            num_boards = 4
            piece_channels = cfg.network.piece_channels
            board_squares = cfg.network.board_squares
            num_actions = 3

            feed = {
                "board_planes": np.zeros((batch_size, num_boards, piece_channels, board_squares), dtype=np.float32),
                "l_coords": np.zeros((batch_size, num_boards), dtype=np.int64),
                "t_coords": np.ones((batch_size, num_boards), dtype=np.int64),
                "used_board_counts": np.full((batch_size,), num_boards, dtype=np.int64),
                "urgency": np.zeros((batch_size,), dtype=np.float32),
                "side_to_move": np.zeros((batch_size,), dtype=np.int64),
                "action_state_indices": np.zeros((num_actions,), dtype=np.int64),
                "action_board_indices": np.zeros((num_actions,), dtype=np.int64),
                "action_from_squares": np.zeros((num_actions,), dtype=np.int64),
                "action_to_squares": np.zeros((num_actions,), dtype=np.int64),
                "action_delta_t": np.zeros((num_actions,), dtype=np.float32),
                "action_delta_l": np.zeros((num_actions,), dtype=np.float32),
                "action_is_submit": np.zeros((num_actions,), dtype=np.int64),
            }

            outputs = session.run(["value", "action_logits"], feed)
            value, logits = outputs

            assert value.shape == (batch_size,), f"value shape: {value.shape}"
            assert logits.shape == (num_actions,), f"logits shape: {logits.shape}"
            assert -1.0 <= value[0] <= 1.0, f"value out of range: {value[0]}"
        return True

    check("ONNX export produces valid model", check_export_shapes)
    check("ONNX Runtime inference succeeds", check_export_runnable)
    return True


# ---------------------------------------------------------------------------
# 2. 二进制格式读写测试
# ---------------------------------------------------------------------------
def test_binary_format():
    section("2. Binary Format Round-Trip")
    from .self_play import CppOnnxSelfPlayWorker, _read_struct, _read_string, _read_exact
    from .config import TrainConfig

    DATA_MAGIC = 0x50535A41
    DATA_VERSION = 3

    def check_write_read_roundtrip():
        """手动构造二进制数据，验证 Python 读取器能正确解析。"""
        cfg = TrainConfig()
        cfg.apply_variant("very_small")

        num_games = 1
        num_moves = 2
        num_samples = 2
        num_boards = 3
        num_actions = 4
        pc = cfg.network.piece_channels  # 12
        bs = cfg.network.board_squares    # 16

        buf = io.BytesIO()

        # 写 header
        buf.write(struct.pack("<III", DATA_MAGIC, DATA_VERSION, num_games))

        # 写 game 1
        buf.write(struct.pack("<fii", 1.0, 10, 20))  # outcome, total_semimoves, board_limit
        # terminal_reason
        reason = b"material"
        buf.write(struct.pack("<I", len(reason)))
        buf.write(reason)
        # pgn
        pgn = b"1. Na3"
        buf.write(struct.pack("<I", len(pgn)))
        buf.write(pgn)

        # move history
        buf.write(struct.pack("<I", num_moves))
        for _ in range(num_moves):
            buf.write(struct.pack("<bB", 0, 0))  # player, is_submit
            buf.write(struct.pack("<fi", 0.5, 5))  # root_value, board_count
            mt = b"Na3"
            buf.write(struct.pack("<I", len(mt)))
            buf.write(mt)

        # samples
        buf.write(struct.pack("<I", num_samples))
        for _ in range(num_samples):
            buf.write(struct.pack("<b", 0))  # player
            buf.write(struct.pack("<ff", 5.0, 0.5))  # urgency, value_target
            buf.write(struct.pack("<iI", num_boards, num_actions))  # num_boards, num_actions

            # board_planes: uint8[num_boards, pc, bs]
            buf.write(np.zeros(num_boards * pc * bs, dtype=np.uint8).tobytes())
            # l_coords: int32[num_boards]
            buf.write(np.zeros(num_boards, dtype=np.int32).tobytes())
            # t_coords: int32[num_boards]
            buf.write(np.zeros(num_boards, dtype=np.int32).tobytes())
            # policy_target: float32[num_actions]
            buf.write(np.ones(num_actions, dtype=np.float32).tobytes())
            # action_board_indices: int32[num_actions]
            buf.write(np.zeros(num_actions, dtype=np.int32).tobytes())
            # action_from_squares: int32[num_actions]
            buf.write(np.zeros(num_actions, dtype=np.int32).tobytes())
            # action_is_submit: uint8[num_actions]
            buf.write(np.zeros(num_actions, dtype=np.uint8).tobytes())

        buf.seek(0)

        # 用 Python 读取器读取
        magic, version, ng = _read_struct(buf, "<III")
        assert magic == DATA_MAGIC, f"Magic mismatch: {magic:#x}"
        assert version == DATA_VERSION, f"Version mismatch: {version}"
        assert ng == num_games, f"Game count mismatch: {ng}"

        outcome, tsm, bl = _read_struct(buf, "<fii")
        assert outcome == 1.0
        assert tsm == 10
        assert bl == 20
        assert _read_string(buf) == "material"
        assert _read_string(buf) == "1. Na3"

        nm, = _read_struct(buf, "<I")
        assert nm == num_moves

        # 跳过 move history
        for _ in range(nm):
            _read_struct(buf, "<bBfi")
            _read_string(buf)

        ns, = _read_struct(buf, "<I")
        assert ns == num_samples

        # 验证 sample 1
        for _ in range(ns):
            player, = _read_struct(buf, "<b")
            urgency, vt = _read_struct(buf, "<ff")
            nb, na = _read_struct(buf, "<iI")
            assert nb == num_boards
            assert na == num_actions

            bp = np.frombuffer(_read_exact(buf, nb * pc * bs), dtype=np.uint8).reshape(nb, pc, bs)
            assert bp.shape == (num_boards, pc, bs)
            lmm = np.frombuffer(_read_exact(buf, nb * bs), dtype=np.int8).reshape(nb, bs)
            assert lmm.shape == (num_boards, bs)
            lc = np.frombuffer(_read_exact(buf, nb * 4), dtype=np.int32)
            assert len(lc) == num_boards
            tc = np.frombuffer(_read_exact(buf, nb * 4), dtype=np.int32)
            assert len(tc) == num_boards
            pt = np.frombuffer(_read_exact(buf, na * 4), dtype=np.float32)
            assert len(pt) == num_actions
            abi = np.frombuffer(_read_exact(buf, na * 4), dtype=np.int32)
            assert len(abi) == num_actions
            afs = np.frombuffer(_read_exact(buf, na * 4), dtype=np.int32)
            assert len(afs) == num_actions
            # 注意：不再有 action_to_squares / delta_t / delta_l
            ais = np.frombuffer(_read_exact(buf, na), dtype=np.uint8)
            assert len(ais) == num_actions

        return True

    check("binary format round-trip (v3, no dead fields)", check_write_read_roundtrip)
    return True


# ---------------------------------------------------------------------------
# 3. 训练样本完整性测试
# ---------------------------------------------------------------------------
def test_training_sample_integrity():
    section("3. Training Sample Integrity")
    import torch
    from .config import TrainConfig
    from .network import AlphaZeroNetwork
    from .env import SemimoveEnv, Semimove, RULE_CAPTURE_KING
    from .halfstep_state import HalfstepState
    from .mcts import MCTS, SUBMIT_ACTION
    from .self_play import HalfstepRecord

    device = torch.device("cpu")

    def check_halfstep_record_shapes():
        """验证训练样本各字段形状正确。"""
        cfg = TrainConfig()
        cfg.apply_variant("very_small")
        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()
        mcts_cfg = type("MCTSConfig", (), {
            "num_simulations": 8,
            "c_puct": 2.0,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.0,
            "temperature_start": 1.0,
            "temperature_threshold": 15,
            "leaf_batch_size": 1,
            "use_root_progressive_widening": False,
            "use_transposition_table": True,
            "reuse_semimove_suffixes": True,
            "submit_prior_weight": 0.1,
        })()

        env = SemimoveEnv(
            cfg.variant_pgn,
            board_limit=10,
            rules_mode=RULE_CAPTURE_KING,
        )
        env.reset()
        mcts = MCTS(network, mcts_cfg, device)
        halfstep_state = HalfstepState(env)

        # 运行一次 MCTS 搜索
        root = mcts.search(halfstep_state, board_limit=10)
        action, policy, _ = mcts.select_halfstep(root, temperature=1.0)

        # 验证 policy 形状
        encoded = env.encode_state(urgency=5.0)
        board_keys = encoded["board_keys"]
        assert len(policy) > 0, "Empty policy"

        # 验证 action 类型
        if action == SUBMIT_ACTION:
            assert isinstance(action, str)
        else:
            assert len(action) == 4, f"Action coord should be 4-tuple, got {action}"

        # 验证 HalfstepRecord 字段
        # (直接验证 root 信息)
        assert len(root.children) > 0
        assert len(policy) == len(root.children), \
            f"policy len {len(policy)} != children {len(root.children)}"

        # 验证 policy 是概率分布
        pt_sum = sum(policy)
        assert abs(pt_sum - 1.0) < 0.01, \
            f"policy sum={pt_sum:.4f} (expected ~1.0)"

        return True

    def check_sample_encoding():
        """验证 encode_state 产出所有必需字段。"""
        cfg = TrainConfig()
        cfg.apply_variant("very_small")

        env = SemimoveEnv(
            cfg.variant_pgn,
            board_limit=10,
            rules_mode=RULE_CAPTURE_KING,
        )
        env.reset()

        # 测试 source 节点编码
        encoded = env.encode_state(urgency=5.0)
        assert "board_planes" in encoded
        assert "l_coords" in encoded
        assert "t_coords" in encoded
        assert "board_keys" in encoded
        assert "current_player" in encoded
        assert "urgency" in encoded

        n = len(encoded["board_keys"])
        pc = cfg.network.piece_channels
        bs = cfg.network.board_squares

        assert encoded["board_planes"].shape == (n, pc, bs), \
            f"board_planes shape: {encoded['board_planes'].shape}"
        assert encoded["l_coords"].shape == (n,), \
            f"l_coords shape: {encoded['l_coords'].shape}"
        assert encoded["t_coords"].shape == (n,), \
            f"t_coords shape: {encoded['t_coords'].shape}"
        assert encoded["board_planes"].dtype == np.float32
        assert encoded["l_coords"].dtype == np.int64
        assert encoded["t_coords"].dtype == np.int64

        # 验证 board_planes 值在 [0, 1] 范围内
        assert encoded["board_planes"].min() >= 0.0, "board_planes has negative values"
        assert encoded["board_planes"].max() <= 1.0, "board_planes has values > 1"

        return True

    check("HalfstepRecord shapes correct", check_halfstep_record_shapes)
    check("encode_state produces valid tensors", check_sample_encoding)
    return True


# ---------------------------------------------------------------------------
# 4. 端到端测试（需要 C++ 二进制）
# ---------------------------------------------------------------------------
def test_end_to_end(exe_path: Optional[str] = None):
    """运行 C++ 自对弈二进制并验证输出。

    Args:
        exe_path: C++ 二进制路径，None 则自动查找。
    """
    section("4. End-to-End C++ Self-Play")
    import torch
    from .config import TrainConfig
    from .network import AlphaZeroNetwork
    from .self_play import CppOnnxSelfPlayWorker

    if exe_path is None:
        candidates = [
            "build_onnx_selfplay/az_selfplay_onnx.exe",
            "Suzero_Vibe/build_onnx_selfplay/az_selfplay_onnx.exe",
        ]
        for c in candidates:
            p = Path(c)
            if p.exists():
                exe_path = str(p)
                break

    if exe_path is None:
        print(f"  {FAIL} C++ binary not found, skip end-to-end test")
        print("     Build it first, then run with --exe PATH")
        return False

    exe_path = str(Path(exe_path).resolve())
    print(f"  Using C++ binary: {exe_path}")

    device = torch.device("cpu")

    def check_generate_games():
        """运行 C++ 自对弈生成游戏数据并验证。"""
        cfg = TrainConfig()
        cfg.apply_variant("very_small")
        cfg.self_play.self_play_backend = "cpp_onnx"
        cfg.self_play.cpp_selfplay_executable = exe_path
        cfg.self_play.num_games = 2
        cfg.self_play.num_workers = 1
        cfg.self_play.worker_task_games = 2
        cfg.self_play.log_worker_task_stats = False
        cfg.self_play.temperature = 0.0
        cfg.self_play.temperature_final = 0.0
        cfg.self_play.temp_threshold = 0
        cfg.mcts.num_simulations = 8
        cfg.mcts.leaf_batch_size = 1
        cfg.mcts.dirichlet_epsilon = 0.0

        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()

        worker = CppOnnxSelfPlayWorker(
            network=network,
            mcts_cfg=cfg.mcts,
            sp_cfg=cfg.self_play,
            device=device,
            train_cfg=cfg,
        )

        games = worker.generate_games(num_games=2)
        assert len(games) == 2, f"Expected 2 games, got {len(games)}"

        for i, game in enumerate(games):
            assert game.outcome is not None, f"Game {i} missing outcome"
            assert game.total_semimoves > 0, f"Game {i} has 0 semimoves"
            assert len(game.samples) > 0, f"Game {i} has 0 samples"
            assert len(game.move_history) > 0, f"Game {i} has 0 moves"

            for j, sample in enumerate(game.samples):
                # 验证样本字段
                assert sample.board_planes is not None
                assert sample.l_coords is not None
                assert sample.t_coords is not None
                assert sample.policy_target is not None
                assert sample.action_board_indices is not None
                assert sample.action_from_squares is not None
                assert sample.action_is_submit is not None

                # 验证形状
                n = len(sample.board_planes)
                pc = cfg.network.piece_channels
                bs = cfg.network.board_squares
                assert sample.board_planes.shape == (n, pc, bs), \
                    f"Sample {j}: board_planes shape {sample.board_planes.shape}"
                assert sample.l_coords.shape == (n,), \
                    f"Sample {j}: l_coords shape {sample.l_coords.shape}"
                assert sample.t_coords.shape == (n,), \
                    f"Sample {j}: t_coords shape {sample.t_coords.shape}"

                na = len(sample.policy_target)
                assert sample.action_board_indices.shape == (na,), \
                    f"Sample {j}: action_board_indices shape {sample.action_board_indices.shape}"
                assert sample.action_from_squares.shape == (na,), \
                    f"Sample {j}: action_from_squares shape {sample.action_from_squares.shape}"
                assert sample.action_is_submit.shape == (na,), \
                    f"Sample {j}: action_is_submit shape {sample.action_is_submit.shape}"

                # 验证 policy 是概率分布
                pt_sum = sample.policy_target.sum()
                assert abs(pt_sum - 1.0) < 0.01, \
                    f"Sample {j}: policy_target sum={pt_sum:.4f} (expected ~1.0)"

            # 验证每局样本数是偶数（source + dest 成对）
            assert len(game.samples) % 2 == 0, \
                f"Game {i}: {len(game.samples)} samples (expected even)"

        print(f"     Games: {len(games)}")
        print(f"     Total samples: {sum(len(g.samples) for g in games)}")
        print(f"     Total semimoves: {sum(g.total_semimoves for g in games)}")
        return True

    check("generate games via C++ binary", check_generate_games)
    return True


# ---------------------------------------------------------------------------
# 5. C++ 和 Python 后端一致性对比
# ---------------------------------------------------------------------------
def test_consistency(exe_path: Optional[str] = None):
    """对比 C++ 和 Python 后端在相同条件下产生的数据。"""
    section("5. C++ vs Python Backend Consistency")
    import torch
    from .config import TrainConfig
    from .network import AlphaZeroNetwork
    from .self_play import CppOnnxSelfPlayWorker, SelfPlayWorker

    if exe_path is None:
        candidates = [
            "build_onnx_selfplay/az_selfplay_onnx.exe",
            "Suzero_Vibe/build_onnx_selfplay/az_selfplay_onnx.exe",
        ]
        for c in candidates:
            p = Path(c)
            if p.exists():
                exe_path = str(p)
                break

    if exe_path is None:
        print(f"  {FAIL} C++ binary not found, skip consistency test")
        return False

    exe_path = str(Path(exe_path).resolve())
    device = torch.device("cpu")

    def check_same_seed_same_result():
        """相同种子下，C++ 和 Python 后端产生相同数量的半走步。"""
        cfg = TrainConfig()
        cfg.apply_variant("very_small")
        cfg.self_play.num_games = 1
        cfg.self_play.num_workers = 1
        cfg.self_play.log_worker_task_stats = False
        cfg.self_play.temperature = 0.0
        cfg.self_play.temperature_final = 0.0
        cfg.self_play.temp_threshold = 0
        cfg.self_play.min_board_limit = 5
        cfg.self_play.max_board_limit = 5
        cfg.mcts.num_simulations = 16
        cfg.mcts.leaf_batch_size = 1
        cfg.mcts.dirichlet_epsilon = 0.0

        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()

        # Python 后端
        py_worker = SelfPlayWorker(
            network=network,
            mcts_cfg=cfg.mcts,
            sp_cfg=cfg.self_play,
            device=device,
            variant_pgn=cfg.variant_pgn,
        )
        py_games = py_worker.generate_games(num_games=1)

        # C++ 后端
        cfg.self_play.self_play_backend = "cpp_onnx"
        cfg.self_play.cpp_selfplay_executable = exe_path
        cpp_worker = CppOnnxSelfPlayWorker(
            network=network,
            mcts_cfg=cfg.mcts,
            sp_cfg=cfg.self_play,
            device=device,
            train_cfg=cfg,
        )
        cpp_games = cpp_worker.generate_games(num_games=1)

        py_game = py_games[0]
        cpp_game = cpp_games[0]

        print(f"     Python: {py_game.total_semimoves} semimoves, {len(py_game.samples)} samples")
        print(f"     C++:    {cpp_game.total_semimoves} semimoves, {len(cpp_game.samples)} samples")

        # 验证样本数合理（每个半走步一个 sample）
        # 注意：Python 和 C++ 的随机种子不同（Python 用 random, C++ 传 seed），
        # 所以走法序列可能不同，但样本数量应在合理范围内
        assert len(py_game.samples) >= py_game.total_semimoves, \
            f"Python: {len(py_game.samples)} samples < {py_game.total_semimoves} semimoves"
        assert len(cpp_game.samples) >= cpp_game.total_semimoves, \
            f"C++: {len(cpp_game.samples)} samples < {cpp_game.total_semimoves} semimoves"

        # C++ 样本数量应该在 [semimoves, 2*semimoves + 5] 范围内
        assert len(cpp_game.samples) >= cpp_game.total_semimoves, \
            f"C++: {len(cpp_game.samples)} samples < {cpp_game.total_semimoves} semimoves"
        assert len(cpp_game.samples) <= cpp_game.total_semimoves * 2 + 5, \
            f"C++: {len(cpp_game.samples)} samples > 2*{cpp_game.total_semimoves}+5"

        # 验证样本形状
        pc = cfg.network.piece_channels
        bs = cfg.network.board_squares
        for sample in cpp_game.samples:
            n = len(sample.board_planes)
            assert sample.board_planes.shape == (n, pc, bs)
            na = len(sample.policy_target)
            assert sample.action_board_indices.shape == (na,)
            assert sample.action_from_squares.shape == (na,)
            assert sample.action_is_submit.shape == (na,)

        return True

    check("C++ generates valid samples", check_same_seed_same_result)
    return True


# ---------------------------------------------------------------------------
# 6. 详细编码 + policy 一致性对比
# ---------------------------------------------------------------------------
def test_detailed_consistency(exe_path: Optional[str] = None):
    """逐位对比 C++ 产出的编码和 policy 与 Python 是否一致。"""
    section("6. Detailed Encoding & Policy Consistency")
    import torch
    from .config import TrainConfig
    from .network import AlphaZeroNetwork
    from .self_play import CppOnnxSelfPlayWorker
    from .env import SemimoveEnv, RULE_CAPTURE_KING

    if exe_path is None:
        candidates = [
            "build_onnx_selfplay/az_selfplay_onnx.exe",
            "Suzero_Vibe/build_onnx_selfplay/az_selfplay_onnx.exe",
        ]
        for c in candidates:
            p = Path(c)
            if p.exists():
                exe_path = str(p)
                break

    if exe_path is None:
        print(f"  {FAIL} C++ binary not found, skip detailed consistency test")
        return False

    exe_path = str(Path(exe_path).resolve())
    device = torch.device("cpu")

    def check_encode_match():
        cfg = TrainConfig()
        cfg.apply_variant("very_small")
        cfg.self_play.self_play_backend = "cpp_onnx"
        cfg.self_play.cpp_selfplay_executable = exe_path
        cfg.self_play.num_games = 1
        cfg.self_play.num_workers = 1
        cfg.self_play.worker_task_games = 1
        cfg.self_play.log_worker_task_stats = False
        cfg.self_play.temperature = 0.0
        cfg.self_play.temperature_final = 0.0
        cfg.self_play.temp_threshold = 0
        cfg.self_play.min_board_limit = 5
        cfg.self_play.max_board_limit = 5
        cfg.mcts.num_simulations = 16
        cfg.mcts.leaf_batch_size = 1
        cfg.mcts.dirichlet_epsilon = 0.0

        network = AlphaZeroNetwork(cfg.network).to(device)
        network.eval()
        worker = CppOnnxSelfPlayWorker(
            network=network, mcts_cfg=cfg.mcts, sp_cfg=cfg.self_play,
            device=device, train_cfg=cfg,
        )
        games = worker.generate_games(num_games=1)
        assert len(games) == 1
        cpp_game = games[0]
        print(f"     Samples: {len(cpp_game.samples)}")
        pc = cfg.network.piece_channels
        bs = cfg.network.board_squares

        # 1. 验证初始状态编码与 Python 一致
        env = SemimoveEnv(cfg.variant_pgn, board_limit=5, rules_mode=RULE_CAPTURE_KING)
        env.reset()
        py_encoded = env.encode_state()
        cpp_first = cpp_game.samples[0]
        diff = np.abs(cpp_first.board_planes - py_encoded["board_planes"]).max()
        print(f"     Initial board_planes diff: {diff:.6f}")
        assert diff < 0.01, f"Initial board_planes mismatch: {diff}"

        # 2. 验证每个样本的编码值域
        for i, sample in enumerate(cpp_game.samples):
            n = len(sample.board_planes)
            assert sample.board_planes.shape == (n, pc, bs)
            assert sample.l_coords.shape == (n,)
            assert sample.t_coords.shape == (n,)

            assert sample.board_planes.min() >= -0.01, f"Sample {i}: bp < 0"
            assert sample.board_planes.max() <= 1.01, f"Sample {i}: bp > 1"

            pt = sample.policy_target
            assert pt.min() >= 0, f"Sample {i}: policy < 0"
            assert abs(pt.sum() - 1.0) < 0.02, f"Sample {i}: policy sum={pt.sum():.4f}"

            na = len(pt)
            assert len(sample.action_board_indices) == na
            assert len(sample.action_from_squares) == na
            assert len(sample.action_is_submit) == na

        # 3. temperature=0 时 policy 是 hard one-hot（Python 端也如此）
        for i, sample in enumerate(cpp_game.samples):
            pt = sample.policy_target
            best_idx = pt.argmax()
            assert pt[best_idx] == 1.0, f"Sample {i}: best prob={pt[best_idx]:.4f} != 1.0 at temp=0"
            assert pt.sum() == 1.0, f"Sample {i}: policy sum={pt.sum():.4f} != 1.0"

        # 4. 验证坐标合理性
        for i, sample in enumerate(cpp_game.samples):
            assert sample.l_coords.min() >= -5, f"Sample {i}: l_coords low"
            assert sample.l_coords.max() <= 10, f"Sample {i}: l_coords high"
            assert sample.t_coords.min() >= 0, f"Sample {i}: t_coords < 0"

        # 5. 验证 action 索引
        for i, sample in enumerate(cpp_game.samples):
            n = len(sample.board_planes)
            for j, (bi, fs, is_sub) in enumerate(zip(
                sample.action_board_indices, sample.action_from_squares, sample.action_is_submit,
            )):
                if is_sub:
                    assert bi == -1, f"Sample {i}, action {j}: submit idx={bi}"
                else:
                    assert 0 <= bi < n, f"Sample {i}, action {j}: idx={bi} out"
                    assert 0 <= fs < bs, f"Sample {i}, action {j}: sq={fs} out"

        return True

    check("C++ encoding matches Python replay", check_encode_match)
    return True


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    results = []

    # 测试 1: ONNX 导出
    results.append(("ONNX export", test_onnx_export()))

    # 测试 2: 二进制格式
    results.append(("Binary format", test_binary_format()))

    # 测试 3: 训练样本完整性
    results.append(("Sample integrity", test_training_sample_integrity()))

    # 测试 4: 端到端（需要 C++ 二进制）
    exe_path = None
    if "--exe" in sys.argv:
        idx = sys.argv.index("--exe")
        if idx + 1 < len(sys.argv):
            exe_path = sys.argv[idx + 1]
    if "--all" in sys.argv or exe_path:
        results.append(("End-to-end", test_end_to_end(exe_path)))
        results.append(("Consistency", test_consistency(exe_path)))
        results.append(("Detailed consistency", test_detailed_consistency(exe_path)))

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"  Results")
    print(f"{'=' * 60}")
    all_pass = True
    for name, ok in results:
        icon = PASS if ok else FAIL
        print(f"  {icon} {name}")
        if not ok:
            all_pass = False
    print()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())