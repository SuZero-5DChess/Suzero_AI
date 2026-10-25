"""Export a trained network to ``smallstep_movekey_v1`` ONNX + sidecar."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .network import SuzeroNetwork
from .protocol import (
    ACTION_ENCODING,
    ACTION_INPUT_ENCODING,
    CPP_BINARY_VERSION,
    IO_PROTOCOL,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAMES,
    SEARCH_ALGORITHM,
    VariantSpec,
)


def _dummy_inputs(
    spec: VariantSpec, device: torch.device, batch: int = 2, boards: int = 3, q: int = 4
) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch, boards, spec.piece_channels, spec.board_squares, device=device),
        torch.zeros(batch, boards, dtype=torch.int64, device=device),
        torch.arange(boards, device=device, dtype=torch.int64).view(1, boards).expand(batch, boards).contiguous(),
        torch.full((batch,), boards, dtype=torch.int64, device=device),
        torch.zeros(batch, dtype=torch.float32, device=device),
        torch.tensor([0, 1] * (batch // 2) + [0] * (batch % 2), dtype=torch.int64, device=device),
        torch.zeros(batch, q, 8, dtype=torch.int64, device=device),
    )


def _sidecar(
    network: SuzeroNetwork, spec: VariantSpec, opset: int
) -> dict[str, Any]:
    config = network.config
    return {
        "format": "suzero_5dchess.onnx.v1",
        "io_protocol": IO_PROTOCOL,
        "action_encoding": ACTION_ENCODING,
        "action_input_encoding": ACTION_INPUT_ENCODING,
        "search_algorithm": SEARCH_ALGORITHM,
        "cpp_binary_version": CPP_BINARY_VERSION,
        "architecture": network.architecture,
        "variant": spec.name,
        "variant_pgn": spec.pgn,
        "board_side": spec.board_side,
        "board_squares": spec.board_squares,
        "piece_channels": spec.piece_channels,
        "line_shift": spec.line_shift,
        "d_model": config.d_model,
        "n_heads": config.n_heads,
        "n_layers": config.n_layers,
        "d_ff": config.d_ff,
        "dropout": config.dropout,
        "action_mlp_hidden": config.action_mlp_hidden,
        "action_feature_dim": network.action_feature_dim,
        "parameter_count": network.parameter_count,
        "opset": int(opset),
        "inputs": [
            {"name": name, "dtype": dtype, "shape": shape}
            for name, dtype, shape in (
                (ONNX_INPUT_NAMES[0], "float32", ["B", "N", spec.piece_channels, spec.board_squares]),
                (ONNX_INPUT_NAMES[1], "int64", ["B", "N"]),
                (ONNX_INPUT_NAMES[2], "int64", ["B", "N"]),
                (ONNX_INPUT_NAMES[3], "int64", ["B"]),
                (ONNX_INPUT_NAMES[4], "float32", ["B"]),
                (ONNX_INPUT_NAMES[5], "int64", ["B"]),
                (ONNX_INPUT_NAMES[6], "int64", ["B", "q", 8]),
            )
        ],
        "outputs": [
            {"name": ONNX_OUTPUT_NAMES[0], "dtype": "float32", "shape": ["B", 1]},
            {"name": ONNX_OUTPUT_NAMES[1], "dtype": "float32", "shape": ["B", 1]},
            {"name": ONNX_OUTPUT_NAMES[2], "dtype": "float32", "shape": ["B", "q"]},
        ],
    }


def _verify_onnx(
    path: Path, torch_outputs: tuple[np.ndarray, ...], inputs: tuple[np.ndarray, ...]
) -> None:
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    try:
        import onnxruntime as ort
    except ImportError:
        return

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feeds = {
        name: value
        for name, value in zip(ONNX_INPUT_NAMES, inputs)
    }
    outputs = session.run(list(ONNX_OUTPUT_NAMES), feeds)
    for expected, actual, name in zip(torch_outputs, outputs, ONNX_OUTPUT_NAMES):
        if not np.allclose(expected, actual, atol=1e-4, rtol=1e-4):
            difference = float(np.max(np.abs(expected - actual)))
            raise RuntimeError(
                f"ONNX output {name!r} differs from torch (max abs {difference})"
            )


def export_network(
    network: SuzeroNetwork,
    spec: VariantSpec,
    output_path: str | os.PathLike[str],
    *,
    opset: int = 18,
    verify: bool = True,
) -> Path:
    """Export atomically and write ``<model>.json`` next to it."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    was_training = network.training
    network.eval()
    device = next(network.parameters()).device
    inputs = _dummy_inputs(spec, device)
    dynamic_axes = {
        ONNX_INPUT_NAMES[0]: {0: "B", 1: "N"},
        ONNX_INPUT_NAMES[1]: {0: "B", 1: "N"},
        ONNX_INPUT_NAMES[2]: {0: "B", 1: "N"},
        ONNX_INPUT_NAMES[3]: {0: "B"},
        ONNX_INPUT_NAMES[4]: {0: "B"},
        ONNX_INPUT_NAMES[5]: {0: "B"},
        ONNX_INPUT_NAMES[6]: {0: "B", 1: "q"},
        ONNX_OUTPUT_NAMES[0]: {0: "B"},
        ONNX_OUTPUT_NAMES[1]: {0: "B"},
        ONNX_OUTPUT_NAMES[2]: {0: "B", 1: "q"},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        with torch.no_grad():
            torch_outputs = tuple(
                tensor.to(torch.float32).cpu().numpy()
                for tensor in network(*inputs)
            )
            torch.onnx.export(
                network,
                inputs,
                str(tmp),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(ONNX_INPUT_NAMES),
                output_names=list(ONNX_OUTPUT_NAMES),
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )
        os.replace(tmp, path)
    finally:
        network.train(was_training)

    sidecar = _sidecar(network, spec, opset)
    sidecar_path = Path(str(path) + ".json")
    sidecar_tmp = Path(str(sidecar_path) + ".tmp")
    with sidecar_tmp.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(sidecar_tmp, sidecar_path)

    if verify:
        _verify_onnx(
            path,
            torch_outputs,
            tuple(value.cpu().numpy() for value in inputs),
        )
    return path
