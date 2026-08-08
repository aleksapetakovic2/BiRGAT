"""Synthetic Sentinel-like event generator.

Produces a realistic enterprise event stream (28 days by default) with
injected attack incidents. This module stands in for the future KQL stage:
whatever it emits, KQL must later emit too (same columns, same semantics).

Honesty rules (anti-cheating)
-----------------------------
1. Every feature distribution used by attack steps is also produced by some
   benign behaviour — admins run encoded PowerShell and manage hosts laterally,
   service accounts beacon periodically and move gigabytes, deployment waves
   look like worm propagation, users produce dense network bursts (meetings /
   sync / streaming). There is NO feature that is 1 only for attacks.
2. Attack start times are drawn **proportional to benign activity density** —
   real hands-on-keyboard intruders blend into busy hours. Hour-of-day and
   day-of-week therefore carry no class signal.
3. No unique identifiers enter the feature matrix (see schema.py).
4. A leakage audit in graph.py hard-fails if any single feature separates
   the classes too well (AUC > threshold); the MLP feature-only baseline
   trained after each run is the second line of defence — if it scores high,
   the features (not the topology) carry the signal and the generator must
   be fixed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .schema import (
    CONFIG_KINDS, ETYPE_ID, FILE_ACTIONS, FILE_CATS, HOST_ROLES, PORT_CLASSES,
    PROCESS_CATS, PROTOCOLS, USER_ROLES,
)

MIN_PER_DAY = 1440.0


# --------------------------------------------------------------------------- #
# Column store                                                                #
# --------------------------------------------------------------------------- #
_COLUMNS: Dict[str, str] = {
    "ts": "f8", "etype": "i1", "host": "i4", "user": "i4", "dst_host": "i4",
    "parent": "i8", "signin": "i8", "y": "i1", "incident": "i4",
    # raw feature fields (feature engineering happens in graph.py)
    "auth_success": "i1", "mfa": "i1", "remote": "i1", "privileged": "i1",
    "integrity": "i1", "outbound": "i1", "protocol": "i1", "port": "i4",
    "bytes": "f8", "process_cat": "i1", "file_cat": "i1", "file_action": "i1",
    "config_kind": "i1", "dns_rarity": "f4", "dns_len": "f4",
    "cmd_entropy": "i1", "encoded_cmd": "i1", "argc": "i2",
}
_DEFAULTS = {
    "user": -1, "dst_host": -1, "parent": -1, "signin": -1, "y": 0,
    "incident": -1, "auth_success": 1, "mfa": 0, "remote": 0, "privileged": 0,
    "integrity": 1, "outbound": 0, "protocol": 0, "port": -1, "bytes": 0.0,
    "process_cat": -1, "file_cat": -1, "file_action": -1, "config_kind": -1,
    "dns_rarity": 0.0, "dns_len": 0.0, "cmd_entropy": 1, "encoded_cmd": 0,
    "argc": 0,
}


class EventStore:
    """Append-only columnar store; finalizes to a dict of numpy arrays."""

    def __init__(self) -> None:
        self._cols: Dict[str, list] = {k: [] for k in _COLUMNS}
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def add(self, ts: float, etype: int, host: int, **kw) -> int:
        self._cols["ts"].append(ts)
        self._cols["etype"].append(etype)
        self._cols["host"].append(host)
        for k in _COLUMNS:
            if k in ("ts", "etype", "host"):
                continue
            self._cols[k].append(kw.get(k, _DEFAULTS[k]))
        idx = self.n
        self.n += 1
        return idx

    def finalize(self) -> Dict[str, np.ndarray]:
        out = {}
        for k, dtype in _COLUMNS.items():
            out[k] = np.asarray(self._cols[k], dtype=dtype)
        return out


@dataclass
class EventTable:
    """Finalized event stream + entity attributes."""
    cols: Dict[str, np.ndarray]
    host_role: np.ndarray          # per host id -> HOST_ROLES index
    host_os: np.ndarray            # per host id -> OS_TYPES index
    user_role: np.ndarray          # per user id -> USER_ROLES index
    n_hosts: int = 0
    n_users: int = 0
    days: int = 0

    def __len__(self) -> int:
        return len(self.cols["ts"])


# --------------------------------------------------------------------------- #
# Generator                                                                   #
# --------------------------------------------------------------------------- #
class SentinelEventGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.data.seed)
        d = cfg.data
        self.days = d.days
        self.n_hosts = d.n_hosts
        self.n_users = d.n_users
        self.store = EventStore()
        self._make_entities()

    # ------------------------------------------------------------- entities
    def _make_entities(self) -> None:
        r = self.rng
        # hosts: mostly workstations, some servers of each flavour
        probs = np.array([0.66, 0.10, 0.02, 0.07, 0.08, 0.07])
        self.host_role = r.choice(len(HOST_ROLES), self.n_hosts, p=probs)
        self.host_os = (r.random(self.n_hosts) < 0.12).astype(np.int64)  # 12% linux (servers)
        linux_ok = np.isin(self.host_role, [1, 2, 4, 5])                  # non-workstations
        self.host_os = np.where(linux_ok, self.host_os, 0)

        # users: regular / admin / service
        uroll = r.random(self.n_users)
        self.user_role = np.where(uroll < 0.82, 0, np.where(uroll < 0.92, 1, 2))
        # ~18% of interactive users work evening/night shifts (global teams,
        # support) — enterprise telemetry does not stop at 18:00
        self.user_shift = (r.random(self.n_users) < 0.18) & (self.user_role != 2)
        self.home_host = np.zeros(self.n_users, dtype=np.int64)
        ws = np.where(self.host_role == 0)[0]
        srv = np.where(self.host_role != 0)[0]
        for u in range(self.n_users):
            if self.user_role[u] == 2:                       # service account
                self.home_host[u] = r.choice(srv)
            else:
                self.home_host[u] = r.choice(ws)
        # admin scope / service targets (lateral management sets)
        self.admin_scope: List[np.ndarray] = [np.empty(0, np.int64)] * self.n_users
        self.svc_targets: List[np.ndarray] = [np.empty(0, np.int64)] * self.n_users
        for u in range(self.n_users):
            if self.user_role[u] == 1:
                k = int(r.integers(2, 10))
                self.admin_scope[u] = r.choice(self.n_hosts, size=k, replace=False)
            elif self.user_role[u] == 2:
                k = int(r.integers(1, 8))
                self.svc_targets[u] = r.choice(srv, size=min(k, len(srv)), replace=False)
        self.activity = np.clip(r.lognormal(0.0, 0.4, self.n_users), 0.4, 2.5)

        # external service port pool (benign); attacks draw from the SAME pool
        self._ext_ports = np.array(
            [443, 80, 53, 22, 25, 587, 993, 8080, 8443, 1433, 3306, 8530, 50001])
        self._ext_p = np.array(
            [.42, .18, .08, .03, .02, .03, .03, .06, .04, .02, .02, .02, .05])
        self._int_ports = np.array([445, 3389, 5985, 22, 135, 53])
        self._int_p = np.array([.30, .25, .15, .10, .10, .10])

    # ------------------------------------------------------------- helpers
    def _proc(self, ts, host, user, cat, parent=-1, signin=-1, integrity=1,
              privileged=0.0, encoded_p=0.0, y=0, incident=-1) -> int:
        r = self.rng
        entropy = int(np.clip(r.normal(1.0 + 0.55 * (cat in (3, 4)), 0.8), 0, 3))
        return self.store.add(
            ts=ts, etype=ETYPE_ID["ProcessCreate"], host=host, user=user,
            parent=parent, signin=signin, y=y, incident=incident,
            integrity=integrity, privileged=int(r.random() < privileged),
            process_cat=cat, cmd_entropy=entropy,
            encoded_cmd=int(r.random() < encoded_p),
            argc=int(r.poisson(2)) + 1,
        )

    def _net(self, ts, host, user=-1, dst_host=-1, port=443, protocol=0,
             bytes_=1e4, outbound=1, signin=-1, y=0, incident=-1) -> int:
        return self.store.add(
            ts=ts, etype=ETYPE_ID["NetworkConnection"], host=host, user=user,
            dst_host=dst_host, port=port, protocol=protocol, bytes=bytes_,
            outbound=outbound, signin=signin, y=y, incident=incident)

    def _signin(self, ts, host, user, remote=0, success=1, privileged=0.0,
                y=0, incident=-1) -> int:
        r = self.rng
        return self.store.add(
            ts=ts, etype=ETYPE_ID["SignIn"], host=host, user=user,
            remote=remote, auth_success=success, mfa=int(r.random() < 0.9),
            privileged=int(r.random() < privileged), y=y, incident=incident)

    def _file(self, ts, host, user, cat, action, signin=-1, bytes_=1e4,
              privileged=0.0, y=0, incident=-1) -> int:
        r = self.rng
        return self.store.add(
            ts=ts, etype=ETYPE_ID["FileActivity"], host=host, user=user,
            file_cat=cat, file_action=action, signin=signin, bytes=bytes_,
            privileged=int(r.random() < privileged), y=y, incident=incident)

    def _config(self, ts, host, user, kind, signin=-1, y=0, incident=-1) -> int:
        return self.store.add(
            ts=ts, etype=ETYPE_ID["SystemConfig"], host=host, user=user,
            config_kind=kind, signin=signin, y=y, incident=incident)

    def _dns(self, ts, host, user=-1, rarity=0.2, length=14.0, y=0,
             incident=-1) -> int:
        r = self.rng
        return self.store.add(
            ts=ts, etype=ETYPE_ID["DnsQuery"], host=host, user=user,
            dns_rarity=rarity, dns_len=max(4.0, r.normal(length, 4.0)),
            y=y, incident=incident)

    def _ext_port(self, c2: bool = False) -> int:
        """Benign and C2 traffic draw from the SAME port pool (overlap!)."""
        if c2 and self.rng.random() < 0.7:
            return int(self.rng.choice([443, 8443, 8080, 50001], p=[.5, .2, .15, .15]))
        return int(self.rng.choice(self._ext_ports, p=self._ext_p))

    # ------------------------------------------------------------- benign
    def _benign_user_day(self, u: int, day: int) -> None:
        r = self.rng
        weekday = day % 7
        active_p = 0.92 if weekday < 5 else 0.15
        if r.random() > active_p:
            return
        if self.user_shift[u]:
            t0 = day * MIN_PER_DAY + np.clip(r.normal(19 * 60, 100),
                                             16 * 60, 24.5 * 60)
        else:
            t0 = day * MIN_PER_DAY + np.clip(r.normal(9 * 60, 70),
                                             6.5 * 60, 11.5 * 60)
        host = int(self.home_host[u])
        admin = self.user_role[u] == 1
        svc = self.user_role[u] == 2
        act = float(self.activity[u])

        # -- interactive sign-in (service accounts don't sign in interactively)
        signin = -1
        if not svc:
            signin = self._signin(t0, host, u, remote=int(admin and r.random() < 0.15))

        # -- failed sign-ins + workstation unlocks/re-auths (Windows logs
        #    these as sign-in events too — realistic volume)
        if r.random() < 0.30:
            self._signin(t0 + r.uniform(0, 480), host, u, success=0)
        if not svc:
            for _ in range(int(r.poisson(1.5))):
                self._signin(t0 + r.uniform(0, 540), host, u)

        # -- process chains
        n_proc = int(r.poisson(14 * act)) if not svc else 0
        active_roots: List[int] = []
        for _ in range(n_proc):
            if r.random() < 0.30 or not active_roots:
                if admin:
                    cat = int(r.choice(14, p=[.22, .21, .05, .07, .10, .06, .04,
                                              .03, .04, .02, .04, .08, .02, .02]))
                else:
                    cat = int(r.choice(14, p=[.26, .26, .05, .04, .05, .06, .04,
                                              .03, .03, .02, .03, .07, .02, .04]))
                pidx = self._proc(
                    t0 + r.uniform(0, 540), host, u, cat, signin=signin,
                    integrity=2 if (admin and r.random() < 0.15) else 1,
                    encoded_p=0.18 if admin else 0.02)
                active_roots.append(pidx)
            else:
                parent = int(r.choice(active_roots))
                cat = int(r.choice(14, p=[.05, .05, .01, .16, .18, .05, .12,
                                          .05, .06, .01, .04, .13, .02, .07]))
                self._proc(self.store._cols["ts"][parent] + float(r.exponential(6.0)),
                           host, u, cat, parent=parent, signin=signin,
                           encoded_p=0.18 if admin else 0.02)

        # -- network activity (external services + some internal, which later
        #    yields benign NET_TRIGGER edges — topological overlap with attacks)
        n_net = int(r.poisson(11 * act)) if not svc else 0
        for _ in range(n_net):
            ts = t0 + r.uniform(0, 560)
            if r.random() < 0.22:
                dst = int(r.choice(self.n_hosts))
                port = int(r.choice(self._int_ports, p=self._int_p))
                self._net(ts, host, u, dst_host=dst, port=port,
                          bytes_=float(r.lognormal(11.5, 1.6)), signin=signin)
            else:
                self._net(ts, host, u, port=self._ext_port(),
                          bytes_=float(r.lognormal(9.0, 1.5)), signin=signin)

        # -- dense benign net burst: meetings / cloud sync / streaming.
        #    Deliberately the same shape as attack C2 + exfil bursts so that
        #    burstiness alone cannot separate the classes (anti-cheat rule 1).
        if not svc and r.random() < 0.35:
            tb = t0 + r.uniform(30, 500)
            for _ in range(int(r.integers(4, 11))):
                self._net(tb + r.uniform(0, 25), host, u, port=self._ext_port(),
                          bytes_=float(r.lognormal(10.5, 1.8)), signin=signin)

        # -- dns / file   (browsers & apps issue lots of DNS — realistic volume)
        for _ in range(int(r.poisson(20 * act))):
            rare = r.random() < 0.10
            self._dns(t0 + r.uniform(0, 560), host, u,
                      rarity=float(r.uniform(0.6, 1.0)) if rare else float(r.beta(2, 12)))
        for _ in range(int(r.poisson(5 * act))):
            self._file(t0 + r.uniform(0, 560), host, u,
                       cat=int(r.choice(9, p=[.55, .06, .05, .06, .10, .01, .10, .04, .03])),
                       action=int(r.choice(4, p=[.5, .35, .05, .10])), signin=signin,
                       bytes_=float(r.lognormal(9.0, 2.0)))

        # -- admin lateral management (hard negatives: looks like lateral movement)
        if admin and r.random() < 0.80:
            for _ in range(int(r.poisson(1.6)) + 1):
                scope = self.admin_scope[u]
                if len(scope) == 0:
                    continue
                tgt = int(r.choice(scope))
                t1 = t0 + r.uniform(20, 520)
                mgmt_signin = self._signin(t1, tgt, u, remote=1,
                                           privileged=0.6)
                self._net(t1 - 1.0, host, u, dst_host=tgt,
                          port=int(r.choice([3389, 5985, 22, 445], p=[.4, .3, .2, .1])))
                for _ in range(int(r.integers(2, 7))):
                    cat = int(r.choice([3, 4, 10, 12, 11], p=[.42, .30, .10, .10, .08]))
                    self._proc(t1 + r.uniform(1, 90), tgt, u, cat, signin=mgmt_signin,
                               integrity=2 if r.random() < 0.4 else 1,
                               encoded_p=0.12, privileged=0.35)
                if r.random() < 0.45:
                    self._config(t1 + r.uniform(5, 120), tgt, u,
                                 kind=int(r.integers(0, 3)), signin=mgmt_signin)

    def _benign_service_day(self, u: int, day: int) -> None:
        """Service accounts: periodic jobs hitting several hosts — deliberately
        beacon/worm-shaped hard negatives."""
        r = self.rng
        if r.random() > 0.9:
            return
        targets = self.svc_targets[u]
        if len(targets) == 0:
            return
        period = float(r.uniform(90, 300))                  # 1.5..5 h
        n_fires = int((MIN_PER_DAY - 60) // period)
        backup = r.random() < 0.4                           # big transfers sometimes
        for f in range(n_fires):
            tf = day * MIN_PER_DAY + 30 + f * period + r.uniform(-6, 6)
            for h in r.choice(targets, size=min(len(targets), int(r.integers(2, 5))),
                              replace=False):
                h = int(h)
                self._signin(tf - 0.5, h, u, remote=1)          # network/service logon
                self._proc(tf, h, u, cat=int(r.choice([9, 10, 11], p=[.45, .35, .20])))
                self._file(tf + 1.0, h, u, cat=8, action=1,
                           bytes_=float(r.lognormal(19.0, 2.0)) if backup
                           else float(r.lognormal(11.0, 1.5)))
                if r.random() < 0.5:
                    self._net(tf + 0.5, h, u, port=int(r.choice([445, 5985])),
                              bytes_=float(r.lognormal(10.0, 1.5)))

    def _benign_waves(self, day: int) -> None:
        """Deployment / AV waves: star topologies that mimic worm spread."""
        r = self.rng
        srv = np.where(self.host_role != 0)[0]
        if r.random() < 0.5:                                # deployment wave
            orch = int(r.choice(srv))
            svc_u = int(r.choice(np.where(self.user_role == 2)[0]))
            t0 = day * MIN_PER_DAY + r.uniform(60, 1200)
            size = min(int(r.integers(15, 70)), self.n_hosts - 1)
            for h in r.choice(self.n_hosts, size=size, replace=False):
                th = t0 + r.uniform(0, 40)
                self._net(th, orch, svc_u, dst_host=int(h), port=445,
                          bytes_=float(r.lognormal(15.0, 1.0)))
                self._proc(th + 1.5, int(h), svc_u, cat=6)   # installer
                self._file(th + 2.0, int(h), svc_u, cat=1, action=1,
                           bytes_=float(r.lognormal(15.0, 1.0)))
                if r.random() < 0.4:                         # installers write
                    self._config(th + r.uniform(3, 30), int(h), svc_u, kind=2)
        if r.random() < 0.4:                                # AV / scan wave
            scanner = int(r.choice(srv))
            svc_u = int(r.choice(np.where(self.user_role == 2)[0]))
            t0 = day * MIN_PER_DAY + r.uniform(60, 1300)
            size = min(int(r.integers(20, 90)), self.n_hosts - 1)
            for h in r.choice(self.n_hosts, size=size, replace=False):
                th = t0 + r.uniform(0, 90)
                self._proc(th, int(h), svc_u, cat=9)
                if r.random() < 0.4:
                    self._net(th + 0.5, scanner, svc_u, dst_host=int(h), port=135)

    def _benign_night_jobs(self, day: int) -> None:
        """Nightly background (20:00-06:00): backups, AV/compliance scans,
        config refreshes, batch jobs. Real enterprise EDR never goes quiet;
        without this volume, hour-of-day would separate the classes."""
        r = self.rng
        T = self.days * MIN_PER_DAY
        svc_users = np.where(self.user_role == 2)[0]
        if len(svc_users) == 0:
            return
        night_start = day * MIN_PER_DAY + 20 * 60.0
        night_span = 10 * 60.0                              # 20:00 -> 06:00
        for h in range(self.n_hosts):
            if r.random() > 0.85:
                continue
            su = int(r.choice(svc_users))
            tb = night_start + r.uniform(0, night_span)
            if tb >= T - 5.0:
                continue
            self._signin(tb, h, su, remote=0)                 # service logon
            kind = r.random()
            if kind < 0.40:                                 # backup job
                self._proc(tb, h, su, cat=int(r.choice([9, 10, 11])))
                self._file(tb + r.uniform(1, 20), h, su, cat=4, action=1,
                           bytes_=float(r.lognormal(18.0, 2.0)))
                if r.random() < 0.6:
                    self._net(tb + r.uniform(0.5, 10), h, su, port=445,
                              bytes_=float(r.lognormal(16.0, 1.5)))
            elif kind < 0.75:                               # AV / compliance scan
                self._proc(tb, h, su, cat=9)
                for _ in range(int(r.integers(1, 4))):
                    self._file(tb + r.uniform(1, 40), h, su,
                               cat=int(r.choice([0, 1, 2])), action=0,
                               bytes_=float(r.lognormal(8.0, 1.5)))
            else:                                           # GPO / agent refresh
                self._config(tb, h, su, kind=int(r.integers(0, 5)))

    # ------------------------------------------------------------- attacks
    def _incident_windows(self):
        """Per-split start windows so every split receives incidents."""
        gcfg = self.cfg.graph
        T = self.days * MIN_PER_DAY
        b1 = gcfg.split_train_frac * T
        b2 = (gcfg.split_train_frac + gcfg.split_val_frac) * T
        g = gcfg.split_gap_hours * 60.0
        max_dur = 720.0                       # chains run <= ~12 h (compressed)
        return [(0.0, max(60.0, b1 - max_dur)),
                (b1 + g, max(b1 + g + 60.0, b2 - max_dur)),
                (b2 + g, max(b2 + g + 60.0, T - max_dur))]

    def _build_activity_cdf(self) -> None:
        """CDF of benign event mass in 30-minute bins. Incident start times
        are drawn from it so attacks blend into enterprise activity — time
        of day must not be a giveaway."""
        ts = np.asarray(self.store._cols["ts"], dtype=np.float64)
        T = self.days * MIN_PER_DAY
        self._act_bins = np.arange(0.0, T + 30.0, 30.0)
        hist, _ = np.histogram(ts, bins=self._act_bins)
        hist = hist.astype(np.float64) + 1.0          # smoothing
        self._act_cdf = np.cumsum(hist)

    def _sample_start(self, lo: float, hi: float) -> float:
        """Sample a start time in [lo, hi) proportional to benign density."""
        if hi - lo < 60.0:
            return float(self.rng.uniform(lo, hi))
        bins, cdf = self._act_bins, self._act_cdf
        i0 = int(np.clip(np.searchsorted(bins, lo, side="right") - 1, 0, len(cdf) - 1))
        i1 = int(np.clip(np.searchsorted(bins, hi, side="left") - 1, 0, len(cdf) - 1))
        base = cdf[i0 - 1] if i0 > 0 else 0.0
        mass = cdf[i1] - base
        if mass <= 0:
            return float(self.rng.uniform(lo, hi))
        u = base + self.rng.random() * mass
        b = int(np.searchsorted(cdf, u, side="left"))
        b = min(b, len(bins) - 2)
        t = bins[b] + self.rng.uniform(0.0, 30.0)
        return float(np.clip(t, lo, hi))

    def _attack_events(self, iid: int, split_target: int) -> None:
        r = self.rng
        lo, hi = self.cfg.data.attack_events_range
        # template selection
        tpl = int(r.choice(5, p=[0.30, 0.20, 0.20, 0.20, 0.10]))
        # start time: proportional to benign activity density inside the
        # assigned split window — attacks blend into busy hours, so
        # hour-of-day / weekday carry no class signal
        wlo, whi = self._incident_windows()[split_target]
        t = self._sample_start(wlo, whi)
        ev_start = len(self.store)

        users_r = np.where(self.user_role == 0)[0]
        users_a = np.where(self.user_role == 1)[0]
        ws = np.where(self.host_role == 0)[0]
        srv = np.where(self.host_role != 0)[0]
        web = np.where(self.host_role == 4)[0]
        if len(web) == 0:
            web = srv

        def adv(lo_m=3.0, hi_m=180.0):
            nonlocal t
            t += float(r.uniform(lo_m, hi_m))

        def proc(h, u, cat, parent=-1, signin=-1, integ=None, priv=0.0, enc=0.0):
            return self._proc(t, int(h), int(u), cat, parent=parent, signin=signin,
                              integrity=integ if integ is not None else (2 if r.random() < .4 else 1),
                              privileged=priv, encoded_p=enc, y=1, incident=iid)

        def net_out(h, u, c2=True, big=False, dst=-1, port=None):
            p = port if port is not None else self._ext_port(c2)
            b = float(r.lognormal(17.5, 1.5) if big else r.lognormal(7.5, 1.2))
            return self._net(t, int(h), int(u), dst_host=int(dst), port=int(p),
                             bytes_=b, y=1, incident=iid)

        def net_in(h, u, dst, port=None):
            p = port if port is not None else int(r.choice([445, 5985, 3389, 22],
                                                            p=[.4, .3, .2, .1]))
            return self._net(t, int(h), int(u), dst_host=int(dst), port=p,
                             bytes_=float(r.lognormal(9.0, 1.2)), y=1, incident=iid)

        def signin(h, u, remote=1, priv=0.5):
            return self._signin(t, int(h), int(u), remote=remote,
                                privileged=priv, y=1, incident=iid)

        def fileev(h, u, cat, action, big=False):
            b = float(r.lognormal(16.5, 1.2) if big else r.lognormal(9.5, 1.8))
            return self._file(t, int(h), int(u), cat, action, bytes_=b,
                              privileged=0.6, y=1, incident=iid)

        def conf(h, u, kind=None):
            k = int(r.integers(0, 3)) if kind is None else kind
            return self._config(t, int(h), int(u), kind=k, y=1, incident=iid)

        def dns_burst(h, u, n):
            # SAME rarity/length mixture as benign DNS (rule 1): a few rare
            # domains mixed into mostly-common ones. Burstiness lives in the
            # *count and timing* (graph shape), not in per-event features.
            for _ in range(n):
                rare = r.random() < 0.10
                self._dns(t + float(r.uniform(0, 8)), int(h), int(u),
                          rarity=float(r.uniform(0.6, 1.0)) if rare
                          else float(r.beta(2, 12)),
                          length=14.0, y=1, incident=iid)

        # ---- T1: phishing -> hands-on-keyboard
        if tpl == 0:
            u = int(r.choice(users_r)); h0 = int(self.home_host[u])
            signin(h0, u, remote=0, priv=0.0)
            adv(1, 20);  p1 = proc(h0, u, 2)                       # mail client
            adv(0.5, 6); p2 = proc(h0, u, 4, parent=p1, enc=.3)   # script interp
            adv(0.5, 8); p3 = proc(h0, u, 3, parent=p2, enc=.25, priv=.3)  # shell
            adv(0.5, 5); fileev(h0, u, 6, 1)                       # drop in temp
            adv(1, 10);  net_out(h0, u)                            # C2
            adv(5, 60);  conf(h0, u)                               # persistence
            adv(10, 90)                                            # dwell
            proc(h0, u, int(r.choice([3, 4])), enc=.25)            # fired by persistence
            adv(5, 40)
            dns_burst(h0, u, int(r.integers(3, 7)))                # discovery
            for _ in range(int(r.integers(1, 4))):                 # remote probes
                adv(0.5, 4)
                signin(int(r.choice(ws)), u, remote=1, priv=0.0)
                self.store._cols["auth_success"][-1] = int(r.random() < 0.25)
            adv(5, 90)
            h1 = int(r.choice(ws)); net_in(h0, u, h1)              # lateral
            adv(0.5, 5); s2 = signin(h1, u, remote=1, priv=.5)
            adv(1, 15); proc(h1, u, int(r.choice([3, 4, 12])), signin=s2, enc=.25)
            adv(5, 90)
            if r.random() < 0.7:                                    # credential access
                fileev(h1, u, 5, 0); proc(h1, u, 11, priv=.8, integ=2)
            adv(10, 120)
            fileev(h1, u, 4, 2)                                     # archive collection
            adv(5, 60); net_out(h1, u, big=True)                   # exfil
            h_last, u_last = h1, u

        # ---- T2: exploit public-facing app
        elif tpl == 1:
            hs = int(r.choice(web)); su = int(r.choice(np.where(self.user_role == 2)[0]))
            proc(hs, su, 12, integ=1)                               # web app proc
            adv(1, 15); p2 = proc(hs, su, 3, integ=2, priv=.6)     # shell child
            adv(2, 30); dns_burst(hs, su, int(r.integers(2, 6)))
            adv(10, 120); conf(hs, su)                              # service persistence
            adv(10, 90)
            h1 = int(r.choice(srv)); net_in(hs, su, h1)
            adv(1, 8); s2 = signin(h1, su, remote=1, priv=.7)
            adv(2, 20); proc(h1, su, int(r.choice([3, 4])), signin=s2, enc=.25)
            adv(10, 120); fileev(h1, su, 5, 0)                      # cred store read
            adv(10, 120); fileev(h1, su, 0, 0, big=True)            # data read
            adv(5, 60); net_out(h1, su, big=True)                   # exfil
            h_last, u_last = h1, su

        # ---- T3: valid-account abuse (low footprint)
        elif tpl == 2:
            u = int(r.choice(np.concatenate([users_r, users_a])))
            h0 = int(r.choice(ws))
            signin(h0, u, remote=1, priv=.3)
            adv(5, 60); proc(h0, u, int(r.choice([3, 8, 11])), enc=.25)
            adv(5, 40); dns_burst(h0, u, int(r.integers(2, 6)))
            adv(5, 90)
            for _ in range(int(r.integers(1, 3))):
                fileev(h0, u, int(r.choice([0, 4])), 0)
                adv(3, 30)
            adv(5, 120); net_out(h0, u, big=r.random() < .6)
            h_last, u_last = h0, u

        # ---- T4: beacon + persistence
        elif tpl == 3:
            u = int(r.choice(users_r)); h0 = int(self.home_host[u])
            signin(h0, u, remote=0, priv=0.0)
            adv(2, 30); p1 = proc(h0, u, 4, enc=.3)
            adv(1, 8);  proc(h0, u, 3, parent=p1, enc=.25)
            adv(2, 15); net_out(h0, u)                              # initial callback
            adv(10, 60); conf(h0, u)
            period = float(r.uniform(20, 60))
            for b in range(int(r.integers(3, 5))):                  # beaconing
                adv(period * 0.8, period * 1.2)
                net_out(h0, u)
            adv(5, 90); proc(h0, u, int(r.choice([3, 4])), enc=.25)
            adv(5, 60); dns_burst(h0, u, int(r.integers(2, 5)))
            h_last, u_last = h0, u

        # ---- T5: service lateral movement
        elif tpl == 4:
            su = int(r.choice(np.where(self.user_role == 2)[0]))
            hA = int(self.home_host[su])
            proc(hA, su, 10, priv=.5)
            adv(5, 40); net_out(hA, su)
            for _ in range(int(r.integers(2, 4))):
                adv(5, 60)
                hX = int(r.choice(srv))
                net_in(hA, su, hX, port=445)
                adv(0.5, 3); conf(hX, su, kind=0)                   # service install
                adv(0.5, 4); fileev(hX, su, 1, 1)                   # binary drop
                adv(0.5, 5); proc(hX, su, int(r.choice([10, 11, 3])), priv=.6)
            adv(15, 120); fileev(hA, su, 5, 0)                      # cred access
            adv(10, 90); net_out(hA, su, big=r.random() < .5)
            h_last, u_last = hA, su

        # ---- honour attack_events_range: real incidents span lo..hi
        # observable events. Templates emit ~15-40; extend with follow-on
        # behaviour (extra C2 check-ins, discovery, process/file touches)
        # until the incident reaches at least `lo`. This adds chain length,
        # not per-event distinguishability.
        guard = 0
        while len(self.store) - ev_start < lo and guard < 20:
            guard += 1
            kind = int(r.integers(0, 4))
            if kind == 0:
                adv(5, 60); net_out(h_last, u_last)
            elif kind == 1:
                adv(5, 60); dns_burst(h_last, u_last, int(r.integers(2, 5)))
            elif kind == 2:
                adv(5, 60)
                proc(h_last, u_last, int(r.choice([3, 4, 11])), enc=.25)
            else:
                adv(5, 60)
                fileev(h_last, u_last, int(r.choice([0, 4, 6])),
                       int(r.integers(0, 2)))
            if len(self.store) - ev_start >= hi:
                break

    # ------------------------------------------------------------- driver
    def generate(self) -> EventTable:
        r = self.rng
        for day in range(self.days):
            for u in range(self.n_users):
                if self.user_role[u] == 2:
                    self._benign_service_day(u, day)
                else:
                    self._benign_user_day(u, day)
            self._benign_waves(day)
            self._benign_night_jobs(day)

        # incident start times are drawn proportional to this benign density
        self._build_activity_cdf()

        n_inc = self.cfg.data.n_incidents
        n_val = max(1, round(0.2 * n_inc)) if n_inc >= 3 else 0
        n_test = max(1, round(0.2 * n_inc)) if n_inc >= 3 else 0
        n_train = max(0, n_inc - n_val - n_test)
        targets = [0] * n_train + [1] * n_val + [2] * n_test
        while len(targets) < n_inc:                       # tiny-n edge case
            targets.append(0)
        r.shuffle(targets)
        for iid in range(n_inc):
            self._attack_events(iid, targets[iid])

        cols = self.store.finalize()
        order = np.argsort(cols["ts"], kind="stable")
        cols = {k: v[order] for k, v in cols.items()}
        # parent / signin pointers must follow the reordering
        remap = np.empty(len(order), dtype=np.int64)
        remap[order] = np.arange(len(order))
        for refcol in ("parent", "signin"):
            v = cols[refcol]
            valid = v >= 0
            v2 = np.full_like(v, -1)
            v2[valid] = remap[v[valid]]
            # references must point to earlier events; drop violations
            v2[(v2 >= 0) & (cols["ts"][v2] > cols["ts"] + 1e-6)] = -1
            cols[refcol] = v2
        return EventTable(cols=cols, host_role=self.host_role, host_os=self.host_os,
                          user_role=self.user_role, n_hosts=self.n_hosts,
                          n_users=self.n_users, days=self.days)


def generate_dataset(cfg: Config) -> EventTable:
    return SentinelEventGenerator(cfg).generate()
