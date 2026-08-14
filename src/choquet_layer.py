from itertools import combinations
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoquetInspiredVotingLayer(nn.Module):
    """Trainable Choquet-inspired voting layer with selectable aggregation modes.

    Modes:
    - inspired: the original pairwise non-additive aggregation used by this demo.
    - discrete_2additive: a discrete Choquet-style sorting/difference formula
      with a 2-additive Mobius approximation of the capacity. This mode is an
      approximation because it does not fully enforce capacity monotonicity.
    """

    VALID_MODES = {"inspired", "discrete_2additive"}
    VALID_ABLATION_MODES = {
        "full",
        "no_pairwise",
        "fixed_singleton",
        "fixed_capacity",
        "no_logit_calibration",
    }

    def __init__(
        self,
        num_agents: int,
        num_classes: int,
        pair_scale: float = 0.45,
        mode: str = "inspired",
        discrete_pair_scale: float = 0.3,
        ablation_mode: str = "full",
    ):
        super().__init__()
        mode = (mode or "inspired").strip().lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported CHOQUET_MODE={mode!r}; expected one of {sorted(self.VALID_MODES)}")
        ablation_mode = (ablation_mode or "full").strip().lower()
        if ablation_mode not in self.VALID_ABLATION_MODES:
            raise ValueError(
                f"Unsupported CHOQUET_ABLATION={ablation_mode!r}; "
                f"expected one of {sorted(self.VALID_ABLATION_MODES)}"
            )
        self.mode = mode
        self.ablation_mode = ablation_mode
        self.num_agents = num_agents
        self.num_classes = num_classes
        self.pair_indices: List[Tuple[int, int]] = list(combinations(range(num_agents), 2))
        self.num_pairs = len(self.pair_indices)
        self.pair_scale = pair_scale
        self.discrete_pair_scale = discrete_pair_scale

        pair_left = torch.tensor([i for i, _ in self.pair_indices], dtype=torch.long)
        pair_right = torch.tensor([j for _, j in self.pair_indices], dtype=torch.long)
        self.register_buffer("pair_left_idx", pair_left, persistent=False)
        self.register_buffer("pair_right_idx", pair_right, persistent=False)

        if self.mode == "inspired":
            # single_weight_i = softmax(a_i task_i + b_i sample_i + c_i conf_i + bias_i)
            self.a = nn.Parameter(torch.ones(num_agents) * 0.7)
            self.b = nn.Parameter(torch.ones(num_agents) * 0.7)
            self.c = nn.Parameter(torch.ones(num_agents) * 0.4)
            self.single_bias = nn.Parameter(torch.zeros(num_agents))

            # pair_weight_ij = sigmoid(u_ij task_i task_j + v_ij sample_i sample_j
            #                          + r_ij agreement_ij + m_ij)
            self.u = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
            self.v = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
            self.r = nn.Parameter(torch.ones(self.num_pairs) * 0.35)
            self.pair_bias = nn.Parameter(torch.zeros(self.num_pairs))
        else:
            # Discrete 2-additive Choquet approximation:
            #   C_mu(f) = sum_i [f_sigma(i)-f_sigma(i-1)] * mu({sigma(i),...,sigma(K)})
            #   mu(S) ~= sum_{i in S} m_i + sum_{i<j in S} m_ij
            # single_m is non-negative and sums to 1; pair_m can be positive
            # (synergy) or negative (redundancy), but full monotonicity is only
            # diagnosed, not enforced, in this first approximation.
            self.raw_single_m = nn.Parameter(torch.zeros(num_agents))
            self.raw_pair_m = nn.Parameter(torch.zeros(self.num_pairs))

        # A small global affine calibration helps CrossEntropyLoss optimize the
        # positive score vector without changing the aggregation mechanism.
        self.logit_scale = nn.Parameter(torch.tensor(4.0))
        self.logit_bias = nn.Parameter(torch.zeros(num_classes))

    def _pair_tensors(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        left = torch.stack([x[:, i] for i, _ in self.pair_indices], dim=1)
        right = torch.stack([x[:, j] for _, j in self.pair_indices], dim=1)
        return left, right

    def _mobius_parameters(self) -> Tuple[torch.Tensor, torch.Tensor]:
        single_m = F.softmax(self.raw_single_m, dim=0)
        pair_m = self.discrete_pair_scale * torch.tanh(self.raw_pair_m)
        if self.ablation_mode in {"fixed_singleton", "fixed_capacity"}:
            single_m = torch.full_like(single_m, 1.0 / self.num_agents)
        if self.ablation_mode in {"no_pairwise", "fixed_capacity"}:
            pair_m = torch.zeros_like(pair_m)
        return single_m, pair_m

    def _capacity_from_mask(
        self,
        mask: torch.Tensor,
        single_m: torch.Tensor,
        pair_m: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        # mask: [batch, num_agents]
        mask = mask.to(dtype=single_m.dtype)
        single_capacity = torch.sum(mask * single_m.unsqueeze(0), dim=1)
        if self.num_pairs:
            pair_mask = mask[:, self.pair_left_idx] * mask[:, self.pair_right_idx]
            pair_capacity = torch.sum(pair_mask * pair_m.unsqueeze(0), dim=1)
        else:
            pair_capacity = torch.zeros_like(single_capacity)
        capacity = single_capacity + pair_capacity
        if not normalize:
            return capacity

        full_mask = torch.ones(1, self.num_agents, device=mask.device, dtype=mask.dtype)
        mu_n = self._capacity_from_mask(full_mask, single_m, pair_m, normalize=False).squeeze(0)
        safe_mu_n = torch.where(mu_n.abs() < 1e-6, torch.ones_like(mu_n), mu_n)
        return capacity / safe_mu_n

    def _forward_inspired(
        self,
        agent_probs: torch.Tensor,
        agent_confidences: torch.Tensor,
        task_relevance: torch.Tensor,
        sample_relevance: torch.Tensor,
        use_pairwise: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        single_logits = (
            self.a * task_relevance
            + self.b * sample_relevance
            + self.c * agent_confidences
            + self.single_bias
        )
        single_weights = F.softmax(single_logits, dim=1)
        single_contribution = torch.sum(single_weights.unsqueeze(-1) * agent_probs, dim=1)

        pi, pj = self._pair_tensors(agent_probs)
        task_i, task_j = self._pair_tensors(task_relevance)
        sample_i, sample_j = self._pair_tensors(sample_relevance)
        pair_task = task_i * task_j
        pair_sample = sample_i * sample_j

        # Agreement can also be implemented with cosine similarity. With
        # probability vectors, 1 - mean L1 distance is stable and easy to read.
        agreement = 1.0 - torch.mean(torch.abs(pi - pj), dim=-1)
        pair_logits = self.u * pair_task + self.v * pair_sample + self.r * agreement + self.pair_bias
        pair_weights = torch.sigmoid(pair_logits)

        # Original Choquet-inspired interaction. This is product-based; replacing
        # pi * pj with torch.minimum(pi, pj) gives a min-style fuzzy interaction.
        interaction = pi * pj
        if use_pairwise:
            pair_contribution = self.pair_scale * torch.sum(
                pair_weights.unsqueeze(-1) * interaction, dim=1
            ) / max(1, self.num_pairs)
        else:
            pair_contribution = torch.zeros_like(single_contribution)

        scores = single_contribution + pair_contribution
        details = {
            "single_weights": single_weights,
            "pair_weights": pair_weights,
            "single_contribution": single_contribution,
            "pair_contribution": pair_contribution,
            "agreement": agreement,
            "mode": torch.tensor(0, device=agent_probs.device),
        }
        return scores, details

    def _forward_discrete_2additive(
        self,
        agent_probs: torch.Tensor,
        use_pairwise: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_agents, num_classes = agent_probs.shape
        single_m, pair_m = self._mobius_parameters()
        if not use_pairwise:
            pair_m = torch.zeros_like(pair_m)

        class_scores = []
        class_capacities = []
        for class_idx in range(num_classes):
            f = agent_probs[:, :, class_idx]
            sorted_values, sorted_indices = torch.sort(f, dim=1, descending=False)
            previous = torch.cat(
                [torch.zeros(batch_size, 1, device=f.device, dtype=f.dtype), sorted_values[:, :-1]],
                dim=1,
            )
            diffs = sorted_values - previous

            capacities = []
            for pos in range(num_agents):
                mask = torch.zeros(batch_size, num_agents, device=f.device, dtype=f.dtype)
                upper_indices = sorted_indices[:, pos:]
                mask.scatter_(1, upper_indices, 1.0)
                capacities.append(self._capacity_from_mask(mask, single_m, pair_m, normalize=True))
            capacity_tensor = torch.stack(capacities, dim=1)
            class_score = torch.sum(diffs * capacity_tensor, dim=1)
            class_scores.append(class_score)
            class_capacities.append(capacity_tensor)

        scores = torch.stack(class_scores, dim=1)
        single_weights = single_m.unsqueeze(0).expand(batch_size, -1)
        pair_weights = pair_m.unsqueeze(0).expand(batch_size, -1)
        details = {
            "single_weights": single_weights,
            "pair_weights": pair_weights,
            "single_contribution": scores,
            "pair_contribution": torch.zeros_like(scores),
            "agreement": torch.zeros(batch_size, self.num_pairs, device=agent_probs.device),
            "mobius_single": single_m,
            "mobius_pair": pair_m,
            "sorted_capacities": torch.stack(class_capacities, dim=-1),
            "mode": torch.tensor(1, device=agent_probs.device),
        }
        return scores, details

    @torch.no_grad()
    def monotonicity_diagnostics(self) -> Dict[str, float]:
        """Check finite-set capacity monotonicity for discrete_2additive mode.

        Reports violations of A subset B but mu(A) > mu(B). This diagnostic does
        not enforce monotonicity; it is meant to document whether the learned
        2-additive approximation behaves like a valid Choquet capacity.
        """
        if self.mode != "discrete_2additive":
            return {"checked": 0, "violations": 0, "max_violation": 0.0}
        single_m, pair_m = self._mobius_parameters()
        device = single_m.device
        capacities = {}
        for bits in range(1 << self.num_agents):
            mask_values = [(bits >> idx) & 1 for idx in range(self.num_agents)]
            mask = torch.tensor([mask_values], device=device, dtype=single_m.dtype)
            capacities[bits] = float(self._capacity_from_mask(mask, single_m, pair_m, normalize=True).item())

        checked = 0
        violations = 0
        max_violation = 0.0
        all_bits = range(1 << self.num_agents)
        for a in all_bits:
            for b in all_bits:
                if a != b and (a & b) == a:
                    checked += 1
                    violation = capacities[a] - capacities[b]
                    if violation > 1e-6:
                        violations += 1
                        max_violation = max(max_violation, violation)
        return {
            "checked": checked,
            "violations": violations,
            "max_violation": max_violation,
        }

    def forward(
        self,
        agent_probs: torch.Tensor,
        agent_confidences: torch.Tensor,
        task_relevance: torch.Tensor,
        sample_relevance: torch.Tensor,
        use_pairwise: bool = True,
        return_details: bool = False,
    ):
        if self.mode == "inspired":
            scores, details = self._forward_inspired(
                agent_probs,
                agent_confidences,
                task_relevance,
                sample_relevance,
                use_pairwise=use_pairwise,
            )
        else:
            scores, details = self._forward_discrete_2additive(
                agent_probs,
                use_pairwise=use_pairwise,
            )

        if self.mode == "discrete_2additive" and self.ablation_mode == "no_logit_calibration":
            logits = scores
        else:
            logits = scores * torch.clamp(self.logit_scale, min=0.1, max=20.0) + self.logit_bias
        if not return_details:
            return logits
        return logits, details
