//! Host resource telemetry: CPU utilisation, load average, memory, swap and the
//! CPU package temperature.
//!
//! GPU telemetry alone can be misleading: a machine that offloads part of a
//! model to the CPU, or that runs several llama-server instances, is often
//! limited by the host rather than by the GPU. Everything here is read from
//! `/proc` and `hwmon`, so no external tool has to be installed.
//!
//! Values that a platform (or a single kernel) does not expose are `None`, which
//! the dashboard renders as an em dash instead of a misleading zero.

use std::path::{Path, PathBuf};

/// hwmon chip names whose `temp1_input` is the CPU package temperature.
const CPU_HWMON_NAMES: [&str; 4] = ["coretemp", "k10temp", "zenpower", "cpu_thermal"];

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SystemMetrics {
    /// Busy share of all cores over the last sample interval, 0-100.
    pub cpu_usage: Option<f32>,
    pub cpu_cores: u32,
    /// CPU package temperature in °C, when hwmon exposes one.
    pub cpu_temp: Option<f32>,
    pub load1: Option<f32>,
    pub load5: Option<f32>,
    pub load15: Option<f32>,
    pub mem_total: u64,
    pub mem_used: u64,
    pub swap_total: u64,
    pub swap_used: u64,
}

/// Reads host counters, remembering the previous `/proc/stat` sample so CPU
/// utilisation can be derived from a delta. A single read only yields jiffies
/// accumulated since boot, which would report a lifetime average.
#[derive(Debug, Default)]
pub struct SystemSampler {
    prev_cpu: Option<CpuTimes>,
    hwmon_temp: Option<PathBuf>,
}

impl SystemSampler {
    pub fn sample(&mut self) -> SystemMetrics {
        let mut m = SystemMetrics {
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(0),
            ..SystemMetrics::default()
        };

        if let Some(cur) = read_cpu_times() {
            m.cpu_usage = self.prev_cpu.and_then(|prev| cpu_usage(prev, cur));
            self.prev_cpu = Some(cur);
        }

        if let Some((load1, load5, load15)) = read_loadavg() {
            m.load1 = Some(load1);
            m.load5 = Some(load5);
            m.load15 = Some(load15);
        }

        if let Some(mem) = read_meminfo() {
            m.mem_total = mem.total;
            m.mem_used = mem.used();
            m.swap_total = mem.swap_total;
            m.swap_used = mem.swap_total.saturating_sub(mem.swap_free);
        }

        m.cpu_temp = self.cpu_temp();

        m
    }

    /// Cached hwmon lookup: rescanning `/sys/class/hwmon` on every poll is
    /// wasteful, and the chip backing `coretemp`/`k10temp` does not move while
    /// the machine is up. A read failure clears the cache so a disappearing
    /// node is re-detected on the next sample.
    fn cpu_temp(&mut self) -> Option<f32> {
        if let Some(path) = &self.hwmon_temp {
            if let Some(temp) = read_temp(path) {
                return Some(temp);
            }
            self.hwmon_temp = None;
        }
        let path = find_cpu_hwmon()?;
        let temp = read_temp(&path)?;
        self.hwmon_temp = Some(path);
        Some(temp)
    }
}

#[derive(Debug, Clone, Copy)]
struct CpuTimes {
    /// Total jiffies across all states.
    total: u64,
    /// Jiffies spent idle (including iowait).
    idle: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct MemInfo {
    total: u64,
    available: u64,
    swap_total: u64,
    swap_free: u64,
}

impl MemInfo {
    /// "Used" excludes reclaimable page cache, which is what `MemAvailable`
    /// expresses. Without it (kernels before 3.14) fall back to free+buffers+
    /// cached, the definition `free(1)` used to report.
    fn used(&self) -> u64 {
        self.total.saturating_sub(self.available)
    }
}

/// Busy share between two `/proc/stat` samples.
fn cpu_usage(prev: CpuTimes, cur: CpuTimes) -> Option<f32> {
    let d_total = cur.total.checked_sub(prev.total)?;
    let d_idle = cur.idle.checked_sub(prev.idle)?;
    if d_total == 0 {
        return None;
    }
    let busy = d_total.saturating_sub(d_idle) as f64;
    Some((busy / d_total as f64 * 100.0) as f32)
}

/// Parse the aggregate `cpu` line of `/proc/stat`.
///
/// Fields are user, nice, system, idle, iowait, irq, softirq, steal, guest,
/// guest_nice. Guest time is already counted inside user/nice, so only the
/// first eight are summed. iowait counts as idle here: the CPU is not executing
/// work while it waits on the disk.
fn parse_cpu_line(stat: &str) -> Option<CpuTimes> {
    let line = stat.lines().find(|l| l.starts_with("cpu "))?;
    let values: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .take(8)
        .map(|f| f.parse::<u64>().ok())
        .collect::<Option<Vec<u64>>>()?;
    if values.len() < 5 {
        return None;
    }
    let total = values.iter().sum();
    if total == 0 {
        return None;
    }
    Some(CpuTimes {
        total,
        idle: values[3] + values[4],
    })
}

/// Parse the three load averages from `/proc/loadavg`.
fn parse_loadavg(text: &str) -> Option<(f32, f32, f32)> {
    let mut fields = text.split_whitespace();
    let one = fields.next()?.parse().ok()?;
    let five = fields.next()?.parse().ok()?;
    let fifteen = fields.next()?.parse().ok()?;
    Some((one, five, fifteen))
}

/// Parse `/proc/meminfo`. Values are reported in kB.
fn parse_meminfo(text: &str) -> MemInfo {
    let mut info = MemInfo::default();
    let (mut free, mut buffers, mut cached) = (0u64, 0u64, 0u64);
    for line in text.lines() {
        let Some((key, rest)) = line.split_once(':') else {
            continue;
        };
        let Some(kb) = rest
            .split_whitespace()
            .next()
            .and_then(|v| v.parse::<u64>().ok())
        else {
            continue;
        };
        let bytes = kb * 1024;
        match key {
            "MemTotal" => info.total = bytes,
            "MemAvailable" => info.available = bytes,
            "MemFree" => free = bytes,
            "Buffers" => buffers = bytes,
            "Cached" => cached = bytes,
            "SwapTotal" => info.swap_total = bytes,
            "SwapFree" => info.swap_free = bytes,
            _ => {}
        }
    }
    if info.available == 0 {
        info.available = free + buffers + cached;
    }
    info
}

fn read_temp(path: &Path) -> Option<f32> {
    let raw = std::fs::read_to_string(path).ok()?;
    let milli: f32 = raw.trim().parse().ok()?;
    // hwmon reports millidegrees; a zero reading usually means "sensor present
    // but not populated", which is worse than showing nothing.
    (milli > 0.0).then_some(milli / 1000.0)
}

fn find_cpu_hwmon() -> Option<PathBuf> {
    for entry in std::fs::read_dir("/sys/class/hwmon").ok()?.flatten() {
        let Ok(name) = std::fs::read_to_string(entry.path().join("name")) else {
            continue;
        };
        if CPU_HWMON_NAMES.contains(&name.trim()) {
            let temp = entry.path().join("temp1_input");
            if temp.exists() {
                return Some(temp);
            }
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_cpu_times() -> Option<CpuTimes> {
    parse_cpu_line(&std::fs::read_to_string("/proc/stat").ok()?)
}

#[cfg(not(target_os = "linux"))]
fn read_cpu_times() -> Option<CpuTimes> {
    None
}

#[cfg(target_os = "linux")]
fn read_loadavg() -> Option<(f32, f32, f32)> {
    parse_loadavg(&std::fs::read_to_string("/proc/loadavg").ok()?)
}

#[cfg(not(target_os = "linux"))]
fn read_loadavg() -> Option<(f32, f32, f32)> {
    None
}

#[cfg(target_os = "linux")]
fn read_meminfo() -> Option<MemInfo> {
    Some(parse_meminfo(
        &std::fs::read_to_string("/proc/meminfo").ok()?,
    ))
}

#[cfg(not(target_os = "linux"))]
fn read_meminfo() -> Option<MemInfo> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAT: &str = "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 50 0 50 400 0 0 0 0 0 0\n";

    #[test]
    fn parses_aggregate_cpu_line() {
        let times = parse_cpu_line(STAT).unwrap();
        assert_eq!(times.total, 1000);
        assert_eq!(times.idle, 800);
    }

    #[test]
    fn ignores_per_core_lines() {
        // Only the aggregate "cpu " line is of interest; a per-core "cpu0" line
        // must not be mistaken for it.
        assert!(parse_cpu_line("cpu0 50 0 50 400 0 0 0 0 0 0\n").is_none());
        assert_eq!(parse_cpu_line(STAT).unwrap().total, 1000);
    }

    #[test]
    fn computes_usage_from_delta() {
        let prev = CpuTimes {
            total: 1000,
            idle: 800,
        };
        let cur = CpuTimes {
            total: 2000,
            idle: 1700,
        };
        // 100 busy jiffies out of 1000 = 10%.
        let usage = cpu_usage(prev, cur).unwrap();
        assert!((usage - 10.0).abs() < 0.01);
    }

    #[test]
    fn usage_is_none_without_progress() {
        let same = CpuTimes {
            total: 1000,
            idle: 800,
        };
        assert!(cpu_usage(same, same).is_none());
    }

    #[test]
    fn parses_loadavg() {
        let (one, five, fifteen) = parse_loadavg("36.06 37.28 31.05 14/2421 43291").unwrap();
        assert!((one - 36.06).abs() < 0.01);
        assert!((five - 37.28).abs() < 0.01);
        assert!((fifteen - 31.05).abs() < 0.01);
    }

    #[test]
    fn parses_meminfo_with_available() {
        let text = "MemTotal:       98766216 kB\nMemFree:        34369712 kB\nMemAvailable:   70750596 kB\nBuffers:          100000 kB\nCached:         20000000 kB\nSwapTotal:       8388604 kB\nSwapFree:        8388604 kB\n";
        let info = parse_meminfo(text);
        assert_eq!(info.total, 98_766_216 * 1024);
        assert_eq!(info.available, 70_750_596 * 1024);
        assert_eq!(info.used(), 28_015_620 * 1024);
        assert_eq!(info.swap_total, 8_388_604 * 1024);
        assert_eq!(info.swap_total - info.swap_free, 0);
    }

    #[test]
    fn falls_back_when_mem_available_is_missing() {
        // Pre-3.14 kernels only expose MemFree/Buffers/Cached.
        let text =
            "MemTotal: 1000 kB\nMemFree: 100 kB\nBuffers: 50 kB\nCached: 250 kB\nSwapTotal: 0 kB\n";
        let info = parse_meminfo(text);
        assert_eq!(info.available, 400 * 1024);
        assert_eq!(info.used(), 600 * 1024);
    }

    #[test]
    fn ignores_unknown_lines_and_units() {
        let info = parse_meminfo(
            "MemTotal: 2048 kB\nHugePages_Total: 0\nHugepagesize: 2048 kB\nMalformedLine\n",
        );
        assert_eq!(info.total, 2048 * 1024);
        assert_eq!(info.swap_total, 0);
    }
}
