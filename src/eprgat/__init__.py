"""eprgat — Event-Provenance Relational Graph Attention Network.

A PoC detector that treats *events* (not entities) as graph nodes, so the
model learns incident topography / behaviour instead of memorising that a
specific IP, host or port is "forever malicious" (the classic failure mode of
entity-centric graphs).

Pipeline (synthetic today, KQL-fed tomorrow):

    raw events  ->  provenance graph (events = nodes, typed edges)
                  ->  RGAT encoder + focal loss + neighbourhood sampling
                  ->  node-level "part of an incident" scores

The node-feature and edge-list contract implemented here (see `schema.py`)
is exactly what the future KQL stage will emit via `make-graph`-style joins
over shared entity columns and timespans (see docs/kql_contract.md).
"""

__version__ = "0.1.0"
