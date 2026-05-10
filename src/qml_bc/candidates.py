"""Expand model grids from config files into explicit candidate specs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from itertools import product
import json
from typing import Any

from qml_bc.config import ExperimentConfig


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    model_name: str
    model_type: str
    model_params: dict[str, Any]
    preprocessing: dict[str, Any]
    quantum: dict[str, Any] | None


def expand_candidates(config: ExperimentConfig) -> list[CandidateSpec]:
    """Expand every model grid in an experiment config."""

    candidates: list[CandidateSpec] = []
    for model in config.models:
        model_name = _required_str(model, "name")
        model_type = _required_str(model, "type")
        preprocessing_grid = _expand_mapping(
            model.get("preprocessing", {}),
            list_as_scalar_keys={"angle_range"},
        )
        params_grid = _expand_mapping(model.get("params", {}))
        quantum_config = model.get("quantum")
        quantum_grid = (
            [None] if quantum_config is None else _expand_mapping(quantum_config)
        )

        for preprocessing, params, quantum in product(
            preprocessing_grid,
            params_grid,
            quantum_grid,
        ):
            payload = {
                "model_name": model_name,
                "model_type": model_type,
                "model_params": params,
                "preprocessing": preprocessing,
                "quantum": quantum,
            }
            candidates.append(
                CandidateSpec(
                    candidate_id=_candidate_id(payload),
                    model_name=model_name,
                    model_type=model_type,
                    model_params=params,
                    preprocessing=preprocessing,
                    quantum=quantum,
                )
            )

    return candidates


def _expand_mapping(
    raw: dict[str, Any],
    list_as_scalar_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("grid sections must be mappings")

    list_as_scalar_keys = list_as_scalar_keys or set()
    keys = list(raw)
    values: list[list[Any]] = []

    for key in keys:
        value = raw[key]
        if isinstance(value, list) and key not in list_as_scalar_keys:
            if not value:
                raise ValueError(f"empty grid for key: {key}")
            values.append(value)
        else:
            values.append([value])

    if not keys:
        return [{}]

    return [dict(zip(keys, combination, strict=True)) for combination in product(*values)]


def _required_str(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"model.{key} must be a non-empty string")
    return value


def _candidate_id(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = sha1(encoded.encode("utf-8")).hexdigest()[:10]
    return f"{payload['model_name']}__{digest}"

