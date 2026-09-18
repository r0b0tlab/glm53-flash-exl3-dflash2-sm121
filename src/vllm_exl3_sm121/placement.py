"""GB10 placement planner: COPY / ALIAS / DISK decisions for unified memory.

On GB10, host RAM and device memory are one 121.7 GiB pool (ATS mode). Every
byte given to a resident copy competes with KV cache, scratch, and the page
cache tier. This planner produces an explicit, receipted allocation:

- tables (Engram, n-gram/PLE)      -> DISK by default (file-backed mmap)
- hot weights                     -> COPY first (they are the forced mass)
- draft modules (DFlash2/MTP/DSpark) -> COPY from the remainder, else ALIAS
- anything that cannot be copied  -> ALIAS if aliasable, else fail closed

Allocation semantics on GB10/ATS: the pool is one 121.7 GiB, but the tiers
differ in reclaimability. COPY = device-managed allocation (fast, not
reclaimable). ALIAS = host pages mapped over ATS (zero-copy; reclaimable by
the OS under pressure, small access cost). DISK = file-backed, page cache
only. The planner therefore budgets device (COPY) residency explicitly and
lets ALIAS/DISK absorb the elastic remainder.

Pure logic. The caller feeds measured sizes (receipts) and gets back a plan
with human-readable receipt lines for the serve log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Placement(str, Enum):
    COPY = "copy"      # resident, pinned device-resident copy
    ALIAS = "alias"    # zero-copy over ATS/UVA host mapping
    DISK = "disk"      # file-backed mmap, page-cache tier only


class Kind(str, Enum):
    WEIGHTS = "weights"
    DRAFT = "draft"
    TABLE = "table"
    VISION = "vision"


@dataclass(frozen=True)
class Item:
    name: str
    kind: Kind
    size_gib: float
    aliasable: bool = True


@dataclass(frozen=True)
class Budget:
    """Measured budget inputs (all GiB). Receipts come from the host."""

    mem_available_gib: float
    kv_gib: float
    scratch_gib: float
    os_gib: float
    reserve_gib: float = 2.0            # never hand the last bytes to weights
    page_cache_target_gib: float = 8.0  # kept free for table/expert paging


@dataclass
class Plan:
    assignments: dict[str, Placement] = field(default_factory=dict)
    receipts: list[str] = field(default_factory=list)
    resident_copy_gib: float = 0.0
    alias_gib: float = 0.0
    disk_gib: float = 0.0

    def placement_of(self, name: str) -> Placement:
        return self.assignments[name]


class PlacementError(RuntimeError):
    """Raised when even the aliasable minimum does not fit (fail closed)."""


def _spendable(b: Budget) -> float:
    return (
        b.mem_available_gib
        - b.kv_gib
        - b.scratch_gib
        - b.os_gib
        - b.reserve_gib
        - b.page_cache_target_gib
    )


def decide(items: list[Item], budget: Budget) -> Plan:
    plan = Plan()
    spendable = _spendable(budget)
    plan.receipts.append(
        f"budget: available={budget.mem_available_gib:.1f} kv={budget.kv_gib:.1f} "
        f"scratch={budget.scratch_gib:.1f} os={budget.os_gib:.1f} "
        f"reserve={budget.reserve_gib:.1f} page_cache={budget.page_cache_target_gib:.1f} "
        f"-> spendable={spendable:.1f} GiB"
    )

    remaining = spendable

    # 1) Tables never take resident copy; they live on disk.
    for it in sorted(items, key=lambda x: -x.size_gib):
        if it.kind is Kind.TABLE:
            plan.assignments[it.name] = Placement.DISK
            plan.disk_gib += it.size_gib

    # 2) Hot weights: the forced mass. Copy what fits, alias (or fail) the rest.
    weights = sorted(
        (i for i in items if i.kind in (Kind.WEIGHTS, Kind.VISION)),
        key=lambda x: x.size_gib,
    )
    total_weights = sum(i.size_gib for i in weights)
    if total_weights <= remaining:
        for it in weights:
            plan.assignments[it.name] = Placement.COPY
            plan.resident_copy_gib += it.size_gib
            remaining -= it.size_gib
        plan.receipts.append(
            f"weights: all COPY ({total_weights:.1f} GiB, remaining {remaining:.1f})"
        )
    else:
        # Copy the largest fitting subset per-item is wrong for a tensor mesh;
        # in practice weight mass is per-module. Copy until full, alias rest.
        for it in weights:
            if it.size_gib <= remaining:
                plan.assignments[it.name] = Placement.COPY
                plan.resident_copy_gib += it.size_gib
                remaining -= it.size_gib
            elif it.aliasable:
                plan.assignments[it.name] = Placement.ALIAS
                plan.alias_gib += it.size_gib
            else:
                raise PlacementError(
                    f"{it.name}: {it.size_gib:.1f} GiB weight does not fit and is "
                    f"not aliasable (remaining {remaining:.1f}); reduce bits or add "
                    "a node"
                )
        plan.receipts.append(
            f"weights: mixed placement (remaining {remaining:.1f} GiB)"
        )

    # 3) Drafts last (hot every decode step, small): fit in the remainder or
    #    alias -- mirrors the measured reference placement (weights copied,
    #    drafter aliased).
    for it in sorted(items, key=lambda x: x.size_gib):
        if it.kind is Kind.DRAFT:
            if it.size_gib <= remaining:
                plan.assignments[it.name] = Placement.COPY
                plan.resident_copy_gib += it.size_gib
                remaining -= it.size_gib
                plan.receipts.append(
                    f"{it.name}: COPY {it.size_gib:.1f} GiB "
                    f"(remaining {remaining:.1f})"
                )
            elif it.aliasable:
                plan.assignments[it.name] = Placement.ALIAS
                plan.alias_gib += it.size_gib
                plan.receipts.append(
                    f"{it.name}: ALIAS {it.size_gib:.1f} GiB (budget shortfall)"
                )
            else:
                raise PlacementError(
                    f"{it.name}: {it.size_gib:.1f} GiB draft does not fit and is "
                    f"not aliasable (remaining {remaining:.1f})"
                )

    plan.receipts.append(
        f"plan total: copy={plan.resident_copy_gib:.1f} "
        f"alias={plan.alias_gib:.1f} disk={plan.disk_gib:.1f} GiB"
    )
    return plan
