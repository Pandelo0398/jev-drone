"""Edge execution profile: load it, check the onboard container against it, and
sample what the onboard computer actually uses while the aircraft flies.

See edge_profiles.yaml for what a profile does and does not claim to emulate.
"""
import json, os, subprocess, threading, time

import numpy as np
import yaml

PROFILES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edge_profiles.yaml")


def load_profile(name):
    with open(PROFILES_FILE) as f:
        profiles = yaml.safe_load(f)["edge_profiles"]
    if name not in profiles:
        raise SystemExit(f"unknown edge profile {name!r}; have {sorted(profiles)}")
    return dict(profiles[name], name=name)


def offline_mode():
    return os.environ.get("OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def local_only(profile):
    return offline_mode() or (profile or {}).get("network_mode") == "local_only"


def enforce_network(profile, provider):
    """local_only / OFFLINE: the decision model must be onboard, never a cloud API."""
    if local_only(profile) and not provider.is_local:
        where = getattr(provider, "url", "a cloud API")
        raise SystemExit(f"network_mode=local_only: provider {provider.name!r} at {where} is not "
                         "onboard. Use --decision-provider laya with a local LAYA_URL.")
    return local_only(profile)


def _docker(*args):
    r = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=10)
    return r.stdout.strip() if r.returncode == 0 else None


def container_limits(name):
    """(cpus, mem_gb, gpu_requested) as configured on the running container, or None."""
    out = _docker("inspect", name, "--format", "{{json .HostConfig}}")
    if not out:
        return None
    hc = json.loads(out)
    cpus = hc.get("NanoCpus", 0) / 1e9 or None
    mem = hc.get("Memory", 0) / 2**30 or None
    gpu = bool(hc.get("DeviceRequests"))
    return {"cpus": cpus, "mem_gb": round(mem, 2) if mem else None, "gpu": gpu}


def check_container(profile):
    """Warn (do not fail) if the onboard container does not match the profile."""
    name = profile.get("container")
    lim = container_limits(name) if name else None
    if lim is None:
        print(f"[edge] container {name!r} not found; resource limits NOT verified", flush=True)
        return None
    problems = []
    if profile.get("cpu_cores") and lim["cpus"] != profile["cpu_cores"]:
        problems.append(f"cpus={lim['cpus']} (profile {profile['cpu_cores']})")
    if profile.get("ram_gb") and (lim["mem_gb"] is None or abs(lim["mem_gb"] - profile["ram_gb"]) > 0.01):
        problems.append(f"mem={lim['mem_gb']}GB (profile {profile['ram_gb']}GB)")
    if not lim["gpu"]:
        problems.append("no GPU requested")
    print(f"[edge] {name}: {lim}" + (f"  MISMATCH: {', '.join(problems)}" if problems else "  ok"), flush=True)
    return dict(lim, matches_profile=not problems)


def gpu_memory_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        used, util = out.stdout.strip().splitlines()[0].split(",")
        return float(used), float(util)
    except Exception:
        return None, None


class ResourceMonitor:
    """Samples the onboard container (cgroup v2) and the GPU about once a second.

    GPU memory is whole-device (WSL does not expose per-process usage), so it is
    reported both absolute and as a delta over `gpu_idle_mb`, the reading taken
    with the model NOT loaded, when the caller knows it.
    """

    def __init__(self, container, period_s=1.0, gpu_idle_mb=None):
        self.container, self.period_s, self.gpu_idle_mb = container, period_s, gpu_idle_mb
        self.samples = []
        self._cg = None
        cid = _docker("inspect", container, "--format", "{{.Id}}") if container else None
        if cid:
            for p in (f"/sys/fs/cgroup/system.slice/docker-{cid}.scope",
                      f"/sys/fs/cgroup/docker/{cid}"):
                if os.path.exists(os.path.join(p, "memory.current")):
                    self._cg = p
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read(self, f):
        with open(os.path.join(self._cg, f)) as fh:
            return fh.read()

    def _cpu_usec(self):
        for line in self._read("cpu.stat").splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])

    def start(self):
        self._thread.start()
        return self

    def _run(self):
        last = None
        while not self._stop.is_set():
            s = {"t": time.time()}
            if self._cg:
                try:
                    s["ram_mb"] = int(self._read("memory.current")) / 2**20
                    cpu = self._cpu_usec()
                    if last:
                        s["cpu_cores"] = (cpu - last[1]) / 1e6 / (s["t"] - last[0])
                    last = (s["t"], cpu)
                except OSError:
                    pass
            s["gpu_mem_mb"], s["gpu_util_pct"] = gpu_memory_used_mb()
            self.samples.append(s)
            self._stop.wait(self.period_s)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)
        return self.summary()

    def summary(self):
        def col(k):
            return [s[k] for s in self.samples if s.get(k) is not None]
        ram, cpu, gmem, gutil = col("ram_mb"), col("cpu_cores"), col("gpu_mem_mb"), col("gpu_util_pct")
        out = {"samples": len(self.samples), "cgroup": bool(self._cg)}
        if ram:
            out.update(container_ram_mb_mean=round(np.mean(ram)), container_ram_mb_max=round(max(ram)))
        if cpu:
            out.update(container_cpu_cores_mean=round(float(np.mean(cpu)), 2),
                       container_cpu_cores_p99=round(float(np.percentile(cpu, 99)), 2))
        if gmem:
            out.update(gpu_mem_mb_max=round(max(gmem)), gpu_util_pct_mean=round(float(np.mean(gutil)), 1))
            if self.gpu_idle_mb is not None:
                out["gpu_mem_model_mb"] = round(max(gmem) - self.gpu_idle_mb)
        return out
