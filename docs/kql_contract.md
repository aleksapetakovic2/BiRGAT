# KQL contract: replacing the synthetic generator with Sentinel data

The synthetic generator emits exactly two artifacts, and the KQL stage must
emit the same two artifacts with the same semantics:

1. **Node table** — one row per *event* (never per entity), with the feature
   columns listed in `src/eprgat/schema.py` (`FEATURE_BLOCKS`).
2. **Edge list** — `(src_event_id, dst_event_id, relation)` rows with a time
   delta and optional byte volume, relations being the six strings in
   `schema.RELATIONS`.

Because events are the nodes, the model can only learn *topography and
behaviour* (chains, fan-outs, cross-host correlation patterns). It can never
learn "IP X is malicious forever" the way entity graphs do — an IP only ever
appears as a coarse hashed/shared-category feature, and the same host can be
benign in one incident context and compromised in another.

## Suggested Sentinel tables

| Event type        | Source tables                                                    |
|-------------------|------------------------------------------------------------------|
| SignIn            | `SigninLogs`, `AADNonInteractiveUserSignInLogs`                  |
| ProcessCreate     | `DeviceProcessEvents`                                            |
| NetworkConnection | `DeviceNetworkEvents`                                            |
| FileActivity      | `DeviceFileEvents`                                               |
| SystemConfig      | `DeviceRegistryEvents`, `DeviceEvents` (service/task changes)    |
| DnsQuery          | `DnsEvents`, `DeviceNetworkEvents` (dns actions)                 |

Restrict to incident-relevant timespans (e.g. `TimeGenerated between
(incidentStart - 24h .. incidentEnd + 2h)` per incident), union, and project
the schema columns. Labels come from incident membership; **alert rows must
not be used as features** (that would leak the detection itself).

## Edge construction (make-graph equivalents)

```kql
// PROCESS_CHILD: process lineage on one device
DeviceProcessEvents
| where TimeGenerated > ago(30d)
| project ChildId=ProcessId, ChildTime=TimeGenerated, DeviceId,
          ParentId=InitiatingProcessId, ParentTime=InitiatingProcessCreationTime
| join kind=inner (
    DeviceProcessEvents
    | project EventId=ProcessId, DeviceId, TimeGenerated)
  on $left.ParentId == $right.EventId and $left.DeviceId == $right.DeviceId
| extend Relation="PROCESS_CHILD"

// NET_TRIGGER: a connection followed by activity on the destination device
DeviceNetworkEvents
| where RemoteDeviceName != "" and TimeGenerated > ago(30d)
| project NetEventId=..., SrcDevice=DeviceId, DstDevice=RemoteDeviceName,
          t=TimeGenerated, Bytes=RemoteIPPort... 
| join kind=inner (
    DeviceEventsUnion   // any event type on the destination
    | project DstEventId=..., DeviceId, TimeGenerated)
  on $left.DstDevice == $right.DeviceId
| where $right.TimeGenerated between (t .. t + 10m)
| extend Relation="NET_TRIGGER"

// SESSION_ACTION: SignIn -> actions of the same account on the same device
// HOST_SEQUENCE / USER_SEQUENCE: ordered joins on DeviceId / AccountUpn
//   within a timespan window (use mv-apply + row_number, or make-graph
//   with a timespan predicate)
// CONFIG_TRIGGER: registry/service/scheduled-task change -> later process
//   creation on the same device within 5m..24h
```

Whatever join you use, cap the per-(source event, relation) out-degree
(default 4, `graph.max_out_degree_per_rel`) and emit the edge features
`dt_minutes` and `bytes` — the model consumes them through the temporal
attention bias.

## Feature columns (KQL must project these)

See `FEATURE_BLOCKS` in `schema.py` for the exact order/widths. Everything is
either a one-hot over a **shared vocabulary** (process category, file
category, port class, …) or a log-scaled statistic — no raw IPs, hostnames,
account names or literal command lines. In KQL, hash high-cardinality values
into buckets (e.g. `hash_md5(...)` then modulo) only where the schema has a
bucketed slot; unique identities stay out of the model by design.

**Do not project windowed aggregates as node features.** Counts like
"events on this host in the last 5 minutes" or "host switches by this
account in the last 15 minutes" are the *topology* the RGAT is supposed to
learn from the edges; emitting them as per-event columns flattens the graph
into the features and lets a flat baseline solve the task on its own. If you
need them later for another consumer, keep them out of the node matrix that
feeds this model.

## Splits

Keep the temporal split rule: train on the earliest ~65% of the timeline,
validate on the next ~17%, test on the rest, with gap zones between them and
**no edges crossing split boundaries**. Evaluating on "the future" is the
only honest evaluation for a detector.
