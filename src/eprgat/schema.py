"""Schema: the contract between the data stage and the model.

This module is the single source of truth for:

* the **event types** (node kinds) — one node per *event*, never per entity;
* the **relation types** (typed, directed provenance edges);
* the **node-feature layout** (named blocks with fixed offsets);
* the **edge-feature layout**.

In the final iteration the KQL stage replaces the synthetic generator and
must emit exactly this contract: an edge list (src_event_id, dst_event_id,
relation) built with ``make-graph``-style joins over shared entity columns /
timespans, plus the node feature matrix below. See ``docs/kql_contract.md``.

Anti-memorisation principle
---------------------------
No *unique* identifier (IP address, hostname, account name, port number) is
ever fed to the model raw. Only coarse, shared-vocabulary categories and
behavioural statistics are used, so the model can only succeed by learning
the *topography and behaviour* of event chains — it can never learn
"port 8443 is malicious" or "host WS-217 is malicious".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------- #
# Event (node) types                                                          #
# --------------------------------------------------------------------------- #
EVENT_TYPES: List[str] = [
    "SignIn",
    "ProcessCreate",
    "NetworkConnection",
    "FileActivity",
    "SystemConfig",
    "DnsQuery",
]
ETYPE_ID: Dict[str, int] = {t: i for i, t in enumerate(EVENT_TYPES)}

# --------------------------------------------------------------------------- #
# Relation (edge) types — directed provenance relations                       #
# --------------------------------------------------------------------------- #
REL_PROCESS_CHILD = "PROCESS_CHILD"      # parent ProcessCreate -> child ProcessCreate (same host)
REL_SESSION_ACTION = "SESSION_ACTION"    # SignIn -> action performed inside that session
REL_HOST_SEQUENCE = "HOST_SEQUENCE"      # event -> next event on the same host (<= window)
REL_USER_SEQUENCE = "USER_SEQUENCE"      # event -> next event of the same user (<= window)
REL_NET_TRIGGER = "NET_TRIGGER"          # NetworkConnection -> correlated event on the dst host
REL_CONFIG_TRIGGER = "CONFIG_TRIGGER"    # SystemConfig -> ProcessCreate it later spawns

RELATIONS: List[str] = [
    REL_PROCESS_CHILD,
    REL_SESSION_ACTION,
    REL_HOST_SEQUENCE,
    REL_USER_SEQUENCE,
    REL_NET_TRIGGER,
    REL_CONFIG_TRIGGER,
]
REL_ID: Dict[str, int] = {r: i for i, r in enumerate(RELATIONS)}

#: edges that cross host boundaries (the long-range, lateral-movement-relevant ones)
CROSS_HOST_RELS = {REL_NET_TRIGGER}

# --------------------------------------------------------------------------- #
# Categorical vocabularies (shared by benign and malicious generators!)       #
# --------------------------------------------------------------------------- #
PROCESS_CATS: List[str] = [
    "browser", "office", "mail_client", "shell", "script_interpreter",
    "devtool", "installer", "archiver", "net_tool", "security_agent",
    "service_mgr", "system_util", "remote_access", "other",
]
FILE_CATS: List[str] = [
    "document", "executable", "library", "script", "archive",
    "credential_store", "temp", "config", "other",
]
FILE_ACTIONS: List[str] = ["read", "write", "delete", "create"]
CONFIG_KINDS: List[str] = ["service", "scheduled_task", "registry_run", "driver", "account"]
PROTOCOLS: List[str] = ["tcp", "udp", "icmp"]
PORT_CLASSES: List[str] = ["na", "wellknown", "registered", "ephemeral"]
INTEGRITY_LEVELS: List[str] = ["low", "medium", "high"]
HOST_ROLES: List[str] = ["workstation", "server", "dc", "fileserver", "webserver", "dbserver"]
OS_TYPES: List[str] = ["windows", "linux"]
USER_ROLES: List[str] = ["regular", "admin", "service"]

# --------------------------------------------------------------------------- #
# Node-feature layout                                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureBlock:
    name: str
    dim: int
    kind: str = "dense"          # dense | onehot
    categories: Tuple[str, ...] = ()


FEATURE_BLOCKS: List[FeatureBlock] = [
    FeatureBlock("etype_onehot", len(EVENT_TYPES), "onehot", tuple(EVENT_TYPES)),
    FeatureBlock("hour_sin_cos", 2),
    FeatureBlock("dow_sin_cos", 2),
    # coarse identity only — deliberately NO unique ids (see module docstring)
    FeatureBlock("host_role_onehot", len(HOST_ROLES), "onehot", tuple(HOST_ROLES)),
    FeatureBlock("os_onehot", len(OS_TYPES), "onehot", tuple(OS_TYPES)),
    FeatureBlock("user_role_onehot", len(USER_ROLES), "onehot", tuple(USER_ROLES)),
    FeatureBlock("auth_flags", 4),                     # success, mfa, remote, privileged
    FeatureBlock("integrity_onehot", len(INTEGRITY_LEVELS), "onehot", tuple(INTEGRITY_LEVELS)),
    FeatureBlock("net_flags", 2),                      # outbound, internal_dst
    FeatureBlock("protocol_onehot", len(PROTOCOLS), "onehot", tuple(PROTOCOLS)),
    FeatureBlock("port_class_onehot", len(PORT_CLASSES), "onehot", tuple(PORT_CLASSES)),
    FeatureBlock("log_bytes", 1),
    FeatureBlock("process_cat_onehot", len(PROCESS_CATS), "onehot", tuple(PROCESS_CATS)),
    FeatureBlock("file_cat_onehot", len(FILE_CATS), "onehot", tuple(FILE_CATS)),
    FeatureBlock("file_action_onehot", len(FILE_ACTIONS), "onehot", tuple(FILE_ACTIONS)),
    FeatureBlock("config_kind_onehot", len(CONFIG_KINDS), "onehot", tuple(CONFIG_KINDS)),
    FeatureBlock("dns_feats", 2),                      # domain rarity, query length
    FeatureBlock("cmdline_feats", 6),                  # entropy bkt(4), encoded flag, log argc
    # Deliberately ABSENT: windowed behavioural statistics (burst counts,
    # host-switch counts, ...). Those are topology flattened into per-event
    # numbers; the RGAT must learn behaviour from the provenance edges
    # instead of being fed pre-computed versions of it.
]

FEATURE_OFFSETS: Dict[str, Tuple[int, int]] = {}
_off = 0
for _b in FEATURE_BLOCKS:
    FEATURE_OFFSETS[_b.name] = (_off, _off + _b.dim)
    _off += _b.dim
FEATURE_DIM: int = _off

# --------------------------------------------------------------------------- #
# Edge-feature layout                                                         #
# --------------------------------------------------------------------------- #
DT_BUCKETS = ["<1m", "1-5m", "5-30m", "30m-2h", "2-12h", ">12h"]
EDGE_FEATURE_BLOCKS: List[FeatureBlock] = [
    FeatureBlock("log_dt_min", 1),
    FeatureBlock("dt_bucket_onehot", len(DT_BUCKETS), "onehot", tuple(DT_BUCKETS)),
    FeatureBlock("log_bytes", 1),
]
EDGE_FEATURE_DIM: int = sum(b.dim for b in EDGE_FEATURE_BLOCKS)
EDGE_DT_BOUNDARIES_MIN = [1.0, 5.0, 30.0, 120.0, 720.0]  # bucket edges in minutes


def dt_bucket(dt_minutes) -> int:
    """Bucket a time-delta (minutes) for the one-hot edge feature."""
    import numpy as np
    return int(np.searchsorted(np.asarray(EDGE_DT_BOUNDARIES_MIN), dt_minutes, side="right"))


def describe_schema() -> str:
    """Human-readable dump of the contract (printed at startup, saved to runs/)."""
    lines = ["=" * 78, "EVENT-PROVENANCE SCHEMA (KQL contract)", "=" * 78]
    lines.append(f"event types ({len(EVENT_TYPES)}):  " + ", ".join(EVENT_TYPES))
    lines.append(f"relations   ({len(RELATIONS)}):  " + ", ".join(RELATIONS))
    lines.append("-" * 78)
    lines.append(f"{'feature block':<24}{'offset':>8}{'dim':>6}  kind")
    for b in FEATURE_BLOCKS:
        s, e = FEATURE_OFFSETS[b.name]
        lines.append(f"{b.name:<24}{s:>8}{b.dim:>6}  {b.kind}")
    lines.append(f"{'TOTAL node features':<24}{FEATURE_DIM:>14}")
    lines.append(f"edge features: {EDGE_FEATURE_DIM}  (log_dt_min + dt bucket onehot + log_bytes)")
    lines.append("=" * 78)
    return "\n".join(lines)
