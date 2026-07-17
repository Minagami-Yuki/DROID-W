"""V46 cross-modal factor reliability with observability protection."""

from __future__ import annotations

from typing import Dict, Optional

import torch


def _smoothstep(value: torch.Tensor) -> torch.Tensor:
    value = torch.clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def _masked_quantile_score(
    value: torch.Tensor,
    valid: torch.Tensor,
    low_quantile: float,
    high_quantile: float,
) -> torch.Tensor:
    """Normalize each edge by its valid-pixel quantiles."""
    scores = torch.zeros_like(value)
    low_quantile = min(max(float(low_quantile), 0.0), 1.0)
    high_quantile = min(max(float(high_quantile), low_quantile + 1e-3), 1.0)
    for edge in range(value.shape[0]):
        edge_valid = valid[edge]
        samples = value[edge][edge_valid]
        if samples.numel() < 2:
            continue
        low = torch.quantile(samples.float(), low_quantile)
        high = torch.quantile(samples.float(), high_quantile)
        denom = torch.clamp(high - low, min=1e-6)
        scores[edge] = _smoothstep((value[edge] - low) / denom)
    return torch.where(valid, scores, torch.zeros_like(scores))


def graph_observability_metadata(
    ii: torch.Tensor,
    jj: torch.Tensor,
    graph_ii: torch.Tensor,
    graph_jj: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Compute inexpensive graph-redundancy proxies for candidate directed edges."""
    if ii.numel() == 0:
        empty = torch.empty_like(ii, dtype=torch.float)
        return {
            "endpoint_degree": empty,
            "reverse_support": empty,
            "edge_span": empty,
        }

    max_node = int(torch.cat((graph_ii, graph_jj, ii, jj)).max().item()) + 1
    degree = torch.zeros(max_node, device=ii.device, dtype=torch.float)
    degree.scatter_add_(0, graph_ii, torch.ones_like(graph_ii, dtype=torch.float))
    degree.scatter_add_(0, graph_jj, torch.ones_like(graph_jj, dtype=torch.float))
    endpoint_degree = torch.minimum(degree[ii], degree[jj])

    adjacency = torch.zeros((max_node, max_node), device=ii.device, dtype=torch.bool)
    adjacency[graph_ii, graph_jj] = True
    reverse_support = adjacency[jj, ii].float()
    edge_span = torch.abs(ii - jj).float()
    return {
        "endpoint_degree": endpoint_degree,
        "reverse_support": reverse_support,
        "edge_span": edge_span,
    }


def apply_v46_reliability(
    scale: torch.Tensor,
    risk: torch.Tensor,
    edge_residual: torch.Tensor,
    valid: torch.Tensor,
    cfg: dict,
    graph_metadata: Optional[Dict[str, torch.Tensor]] = None,
    prior_risk: Optional[torch.Tensor] = None,
    prior_valid: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Validate semantic risk geometrically and protect graph information.

    The incoming ``scale`` is the existing v25-style candidate attenuation. V46
    only removes attenuation that is unsupported or unsafe; it never creates a
    stronger attenuation than the candidate.
    """
    if not bool(cfg.get("enable", True)):
        identity = scale.clone()
        empty = torch.zeros_like(scale)
        return identity, {
            "prior_score": empty,
            "geometry_score": empty,
            "agreement": torch.ones_like(scale),
            "temporal_score": torch.ones_like(scale),
            "observability": torch.ones_like(scale),
            "endpoint_degree": empty,
            "reverse_support": empty,
            "edge_span": empty,
            "budget_multiplier": torch.ones_like(scale),
            "candidate_scale": identity,
            "output_scale": identity,
        }

    risk = torch.clamp(risk.float(), 0.0, 1.0)
    edge_residual = torch.clamp(edge_residual.float(), 0.0, 1.0)
    valid = valid.bool()
    candidate_scale = torch.clamp(scale.float(), 0.0, 1.0)
    candidate_suppression = 1.0 - candidate_scale

    agreement_cfg = cfg.get("agreement", {}) or {}
    if bool(agreement_cfg.get("enable", True)):
        prior_score = _masked_quantile_score(
            risk,
            valid,
            agreement_cfg.get("risk_low_quantile", 0.50),
            agreement_cfg.get("risk_high_quantile", 0.85),
        )
        geometry_score = _masked_quantile_score(
            edge_residual,
            valid,
            agreement_cfg.get("residual_low_quantile", 0.50),
            agreement_cfg.get("residual_high_quantile", 0.85),
        )
        agreement = torch.pow(
            prior_score * geometry_score,
            float(agreement_cfg.get("power", 1.0)),
        )
        agreement_floor = float(agreement_cfg.get("floor", 0.0))
        agreement = agreement_floor + (1.0 - agreement_floor) * agreement
    else:
        prior_score = risk
        geometry_score = edge_residual
        agreement = torch.ones_like(risk)

    temporal_cfg = cfg.get("temporal", {}) or {}
    temporal_score = torch.ones_like(risk)
    if bool(temporal_cfg.get("enable", False)):
        fallback = float(temporal_cfg.get("first_observation_score", 0.35))
        temporal_score = torch.full_like(risk, fallback)
        if prior_risk is not None and prior_valid is not None:
            history_valid = valid & prior_valid.bool()
            history_score = _masked_quantile_score(
                torch.clamp(prior_risk.float(), 0.0, 1.0),
                history_valid,
                temporal_cfg.get("risk_low_quantile", 0.50),
                temporal_cfg.get("risk_high_quantile", 0.85),
            )
            persistent = torch.sqrt(torch.clamp(prior_score * history_score, min=0.0))
            temporal_score = torch.where(history_valid, persistent, temporal_score)
        temporal_floor = float(temporal_cfg.get("floor", 0.25))
        temporal_score = temporal_floor + (1.0 - temporal_floor) * temporal_score

    observability_cfg = cfg.get("observability", {}) or {}
    edge_count = scale.shape[0]
    device = scale.device
    endpoint_degree = torch.full((edge_count,), 99.0, device=device)
    reverse_support = torch.ones((edge_count,), device=device)
    edge_span = torch.zeros((edge_count,), device=device)
    if graph_metadata is not None:
        endpoint_degree = graph_metadata.get("endpoint_degree", endpoint_degree).float()
        reverse_support = graph_metadata.get("reverse_support", reverse_support).float()
        edge_span = graph_metadata.get("edge_span", edge_span).float()

    observability = torch.ones((edge_count,), device=device)
    if bool(observability_cfg.get("enable", False)):
        min_degree = float(observability_cfg.get("min_endpoint_degree", 2.0))
        full_degree = float(observability_cfg.get("full_endpoint_degree", 5.0))
        degree_score = _smoothstep(
            (endpoint_degree - min_degree) / max(full_degree - min_degree, 1e-6)
        )
        reverse_floor = float(observability_cfg.get("no_reverse_floor", 0.35))
        reverse_score = reverse_floor + (1.0 - reverse_floor) * reverse_support
        observability = degree_score * reverse_score

        if bool(observability_cfg.get("protect_long_span", True)):
            span_start = float(observability_cfg.get("span_protect_start", 4.0))
            span_full = float(observability_cfg.get("span_protect_full", 12.0))
            long_span = _smoothstep(
                (edge_span - span_start) / max(span_full - span_start, 1e-6)
            )
            min_long_span_scale = float(observability_cfg.get("long_span_min_scale", 0.25))
            span_score = 1.0 - (1.0 - min_long_span_scale) * long_span
            observability = observability * span_score

    suppression = candidate_suppression * agreement * temporal_score
    suppression = suppression * observability.view(-1, 1, 1)

    budget_cfg = cfg.get("information_budget", {}) or {}
    budget_multiplier = torch.ones((edge_count,), device=device)
    if bool(budget_cfg.get("enable", False)):
        flat_valid = valid.reshape(edge_count, -1)
        denom = flat_valid.sum(dim=1).clamp(min=1).float()
        mean_suppression = (
            suppression.reshape(edge_count, -1) * flat_valid.float()
        ).sum(dim=1) / denom
        max_mean_suppression = float(budget_cfg.get("max_mean_suppression", 0.003))
        budget_multiplier = torch.clamp(
            max_mean_suppression / torch.clamp(mean_suppression, min=1e-8),
            max=1.0,
        )
        suppression = suppression * budget_multiplier.view(-1, 1, 1)

    output_scale = torch.where(valid, 1.0 - suppression, torch.ones_like(suppression))
    min_scale = float(cfg.get("min_scale", 0.97))
    output_scale = torch.clamp(output_scale, min=min_scale, max=1.0)
    debug = {
        "prior_score": prior_score.detach(),
        "geometry_score": geometry_score.detach(),
        "agreement": agreement.detach(),
        "temporal_score": temporal_score.detach(),
        "observability": observability.detach().view(-1, 1, 1).expand_as(risk),
        "endpoint_degree": endpoint_degree.detach().view(-1, 1, 1).expand_as(risk),
        "reverse_support": reverse_support.detach().view(-1, 1, 1).expand_as(risk),
        "edge_span": edge_span.detach().view(-1, 1, 1).expand_as(risk),
        "budget_multiplier": budget_multiplier.detach().view(-1, 1, 1).expand_as(risk),
        "candidate_scale": candidate_scale.detach(),
        "output_scale": output_scale.detach(),
    }
    return output_scale.to(dtype=scale.dtype), debug
