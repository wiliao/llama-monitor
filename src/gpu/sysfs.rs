use anyhow::Result;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use super::{GpuBackend, GpuMetrics};

/// Root of the DRM device tree.
pub const DRM_ROOT: &str = "/sys/class/drm";

/// PCI vendor ID for AMD/ATI.
const AMD_VENDOR: &str = "0x1002";

const BYTES_PER_MIB: u64 = 1024 * 1024;
const MICROWATTS_PER_WATT: f64 = 1_000_000.0;

/// AMD GPU telemetry read straight from the amdgpu kernel interface.
///
/// This is the same source `rocm-smi` itself reads, so it needs **no ROCm
/// userspace** (no `rocm-smi` binary, no `librocm_smi64`) and no Vulkan either:
/// neither API exposes sensors — Vulkan reports only static device properties,
/// and this covers all of [`GpuMetrics`]. Cards are matched by PCI vendor ID,
/// never by a hardcoded `cardN`: DRM numbering is not stable across reboots or
/// between hosts.
pub struct SysfsBackend;

impl GpuBackend for SysfsBackend {
    fn read_metrics(&self) -> Result<BTreeMap<String, GpuMetrics>> {
        read_amd_metrics(Path::new(DRM_ROOT))
    }

    fn name(&self) -> &str {
        "sysfs"
    }
}

/// True when at least one AMD card is exposed under the DRM root.
pub fn amd_cards_present() -> bool {
    amd_cards_present_in(Path::new(DRM_ROOT))
}

pub fn amd_cards_present_in(drm_root: &Path) -> bool {
    !amd_card_dirs(drm_root).is_empty()
}

/// Every AMD DRM card, sorted by node name for stable ordering.
fn amd_card_dirs(drm_root: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return Vec::new();
    };
    let mut cards: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| is_drm_card(path))
        .filter(|path| read_trimmed(&path.join("device/vendor")).as_deref() == Some(AMD_VENDOR))
        .collect();
    cards.sort();
    cards
}

/// `card0`, `card1`, … but not `card1-DP-1` or `renderD128`.
fn is_drm_card(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix("card"))
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
}

/// Read telemetry for every AMD card under `drm_root`, keyed by display name.
///
/// `drm_root` is a parameter so tests can point at a fixture tree.
pub fn read_amd_metrics(drm_root: &Path) -> Result<BTreeMap<String, GpuMetrics>> {
    let mut metrics: BTreeMap<String, GpuMetrics> = BTreeMap::new();

    for card in amd_card_dirs(drm_root) {
        let Some((base_name, card_metrics)) = read_card(&card) else {
            continue;
        };
        // Keying by display name matches the rocm-smi backend, so the web UI
        // shows "AMD Radeon RX 7900 XTX" rather than "card1". Two identical
        // cards would collide, hence the suffix.
        let name = unique_name(&metrics, &base_name);
        metrics.insert(name, card_metrics);
    }

    Ok(metrics)
}

fn read_card(card_dir: &Path) -> Option<(String, GpuMetrics)> {
    let device = card_dir.join("device");
    let hwmon = first_hwmon_dir(&device);
    let hwmon = hwmon.as_deref();

    let card_metrics = GpuMetrics {
        temp: hwmon.map(read_temp_c).unwrap_or(0.0),
        load: read_u64(&device.join("gpu_busy_percent")).unwrap_or(0) as u32,
        power_consumption: hwmon.map(read_power_w).unwrap_or(0.0),
        power_limit: hwmon.and_then(read_power_cap_w).unwrap_or(0),
        vram_used: read_u64(&device.join("mem_info_vram_used")).unwrap_or(0) / BYTES_PER_MIB,
        vram_total: read_u64(&device.join("mem_info_vram_total")).unwrap_or(0) / BYTES_PER_MIB,
        sclk_mhz: read_file(&device.join("pp_dpm_sclk"))
            .map(|table| parse_active_clock(&table))
            .unwrap_or(0),
        mclk_mhz: read_file(&device.join("pp_dpm_mclk"))
            .map(|table| parse_active_clock(&table))
            .unwrap_or(0),
    };

    Some((card_name(card_dir, &device), card_metrics))
}

fn unique_name(existing: &BTreeMap<String, GpuMetrics>, base: &str) -> String {
    if !existing.contains_key(base) {
        return base.to_string();
    }
    (2..)
        .map(|i| format!("{base} #{i}"))
        .find(|candidate| !existing.contains_key(candidate))
        .unwrap_or_else(|| base.to_string())
}

/// Pick the card's display name: a known marketing name, else `lspci`, else the
/// DRM node name.
fn card_name(card_dir: &Path, device: &Path) -> String {
    if let Some(pci_id) = pci_id(device) {
        if let Some(known) = known_name(&pci_id) {
            return known.to_string();
        }
        if let Some(from_lspci) = lspci_name(device) {
            return from_lspci;
        }
        return format!("AMD GPU {pci_id}");
    }

    card_dir
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("AMD GPU")
        .to_string()
}

/// Marketing names for the few device IDs that are unambiguous enough to label.
///
/// Several of these IDs cover a whole family (e.g. Navi 31 `744c` is the 7900
/// XT/XTX/GRE), so the label is best-effort; anything unknown falls through to
/// `lspci` and then to the DRM node name rather than guessing.
fn known_name(pci_id: &str) -> Option<&'static str> {
    let device = pci_id.split(':').nth(1)?;
    let table: &[(&str, &str)] = &[
        ("744c", "AMD Radeon RX 7900 XTX"),         // Navi 31
        ("73bf", "AMD Radeon RX 6800/6900 series"), // Navi 21
        ("73df", "AMD Radeon RX 6700 series"),      // Navi 22
        ("73ff", "AMD Radeon RX 6600 series"),      // Navi 23
        ("731f", "AMD Radeon RX 5700 series"),      // Navi 10
        ("67df", "AMD Radeon RX 470/480/570/580"),  // Polaris 10
    ];
    table
        .iter()
        .find(|(id, _)| id.eq_ignore_ascii_case(device))
        .map(|(_, name)| *name)
}

/// `PCI_ID=1002:744C` from sysfs `uevent`, normalised to upper-case.
fn pci_id(device: &Path) -> Option<String> {
    let uevent = read_file(&device.join("uevent"))?;
    let line = uevent.lines().find(|l| l.starts_with("PCI_ID="))?;
    let value = line.trim_start_matches("PCI_ID=").trim();
    (!value.is_empty()).then(|| value.to_ascii_uppercase())
}

/// Best-effort marketing name from `pciutils`, used only for unknown IDs.
fn lspci_name(device: &Path) -> Option<String> {
    let uevent = read_file(&device.join("uevent"))?;
    let slot = uevent
        .lines()
        .find_map(|l| l.strip_prefix("PCI_SLOT_NAME="))?
        .trim();

    let output = std::process::Command::new("lspci")
        .args(["-mm", "-s", slot])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let fields = parse_quoted_fields(stdout.lines().next()?);
    // lspci -mm: [slot, class, vendor, device, ...]. The device name is the
    // informative one; the subsystem fields that follow are model-specific.
    fields
        .get(3)
        .map(|name| name.trim())
        .filter(|name| !name.is_empty() && *name != "N/A")
        .map(str::to_string)
}

/// Split an `lspci -mm` line into its quoted fields.
pub fn parse_quoted_fields(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut started = false;

    for ch in line.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                started = true;
            }
            c if c.is_whitespace() && !in_quotes => {
                if started {
                    fields.push(std::mem::take(&mut current));
                    started = false;
                }
            }
            c => {
                current.push(c);
                started = true;
            }
        }
    }
    if started {
        fields.push(current);
    }

    fields
}

/// Frequency of the DPM state the driver has marked active (`*`), in MHz.
///
/// Lines look like `0: 2371Mhz *`, `S: 0Mhz *`, or `1: 500Mhz` (inactive).
/// The last number on the active line is the frequency — the first is the state
/// index.
pub fn parse_active_clock(table: &str) -> u32 {
    let Some(line) = table.lines().find(|line| line.contains('*')) else {
        return 0;
    };
    let before_marker = line.split('*').next().unwrap_or(line);
    before_marker
        .split(|c: char| !c.is_ascii_digit())
        .rfind(|part| !part.is_empty())
        .and_then(|mhz| mhz.parse().ok())
        .unwrap_or(0)
}

/// First hwmon node of a device — the number (`hwmon2`) is not stable.
fn first_hwmon_dir(device: &Path) -> Option<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(device.join("hwmon"))
        .ok()?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs.into_iter().next()
}

/// Junction temperature when available, else edge, else the first sensor.
///
/// `rocm-smi`'s "Sensor junction" is what the rocm backend reports, so prefer
/// the same sensor to keep the dashboard reading stable across backends.
fn read_temp_c(hwmon: &Path) -> f32 {
    let mut sensors: Vec<(String, f32)> = Vec::new();
    for index in 1..=10 {
        let Some(millidegrees) = read_u64(&hwmon.join(format!("temp{index}_input"))) else {
            continue;
        };
        let label = read_trimmed(&hwmon.join(format!("temp{index}_label"))).unwrap_or_default();
        sensors.push((label, millidegrees as f32 / 1000.0));
    }

    let by_label = |want: &str| {
        sensors
            .iter()
            .find(|(label, _)| label == want)
            .map(|(_, value)| *value)
    };

    by_label("junction")
        .or_else(|| by_label("edge"))
        .or_else(|| sensors.first().map(|(_, value)| *value))
        .unwrap_or(0.0)
}

fn read_power_w(hwmon: &Path) -> f32 {
    read_u64(&hwmon.join("power1_average"))
        .or_else(|| read_u64(&hwmon.join("power1_input")))
        .map(|microwatts| (microwatts as f64 / MICROWATTS_PER_WATT) as f32)
        .unwrap_or(0.0)
}

/// Power cap in watts: the configured cap when set, else the hardware maximum.
fn read_power_cap_w(hwmon: &Path) -> Option<u32> {
    let configured = read_u64(&hwmon.join("power1_cap")).filter(|cap| *cap > 0);
    let microwatts = configured.or_else(|| read_u64(&hwmon.join("power1_cap_max")))?;
    Some((microwatts as f64 / MICROWATTS_PER_WATT) as u32)
}

fn read_file(path: &Path) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

fn read_trimmed(path: &Path) -> Option<String> {
    read_file(path).map(|value| value.trim().to_string())
}

fn read_u64(path: &Path) -> Option<u64> {
    read_trimmed(path)?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a throwaway DRM tree. Cards are written as real sysfs does:
    /// `cardN/device/{vendor,uevent,...}` plus `device/hwmon/hwmonX/…`.
    fn fixture_root(tag: &str) -> PathBuf {
        let root =
            std::env::temp_dir().join(format!("llama-monitor-sysfs-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    fn write(path: PathBuf, content: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn write_amd_card(root: &Path, card: &str, device_id: &str, slot: &str) {
        let device = root.join(card).join("device");
        write(device.join("vendor"), AMD_VENDOR);
        write(
            device.join("uevent"),
            &format!("DRIVER=amdgpu\nPCI_ID=1002:{device_id}\nPCI_SLOT_NAME={slot}\n"),
        );
        write(device.join("gpu_busy_percent"), "87\n");
        write(device.join("mem_info_vram_used"), "21455732736\n");
        write(device.join("mem_info_vram_total"), "25753026560\n");
        write(
            device.join("pp_dpm_sclk"),
            "S: 0Mhz *\n1: 500Mhz \n2: 2371Mhz \n",
        );
        write(
            device.join("pp_dpm_mclk"),
            "0: 96Mhz \n1: 456Mhz \n2: 772Mhz \n3: 1249Mhz *\n",
        );

        let hwmon = device.join("hwmon").join("hwmon2");
        write(hwmon.join("temp1_input"), "28000\n");
        write(hwmon.join("temp1_label"), "edge\n");
        write(hwmon.join("temp2_input"), "34000\n");
        write(hwmon.join("temp2_label"), "junction\n");
        write(hwmon.join("temp3_input"), "38000\n");
        write(hwmon.join("temp3_label"), "mem\n");
        write(hwmon.join("power1_average"), "180500000\n");
        write(hwmon.join("power1_cap"), "0\n");
        write(hwmon.join("power1_cap_max"), "303000000\n");
    }

    #[test]
    fn reads_amd_card_from_sysfs() {
        let root = fixture_root("amd");
        write_amd_card(&root, "card1", "744C", "0000:29:00.0");

        let metrics = read_amd_metrics(&root).unwrap();
        assert_eq!(metrics.len(), 1);

        let card = metrics.get("AMD Radeon RX 7900 XTX").unwrap();
        assert!((card.temp - 34.0).abs() < 0.1, "prefers junction over edge");
        assert_eq!(card.load, 87);
        assert!((card.power_consumption - 180.5).abs() < 0.1);
        assert_eq!(card.power_limit, 303);
        assert_eq!(card.vram_used, 20461); // 21455732736 / 1024 / 1024, truncated
        assert_eq!(card.vram_total, 24560);
        assert_eq!(card.sclk_mhz, 0); // idle state is the active one here
        assert_eq!(card.mclk_mhz, 1249);
    }

    #[test]
    fn skips_non_amd_cards_and_connectors() {
        let root = fixture_root("mixed");
        write_amd_card(&root, "card1", "744C", "0000:29:00.0");

        // NVIDIA display card, an Intel iGPU, a connector and a render node.
        write(root.join("card2/device/vendor"), "0x10de\n");
        write(root.join("card0/device/vendor"), "0x8086\n");
        write(root.join("card1-DP-1/device/vendor"), AMD_VENDOR);
        write(root.join("renderD128/device/vendor"), AMD_VENDOR);

        let metrics = read_amd_metrics(&root).unwrap();
        assert_eq!(metrics.len(), 1);
        assert!(metrics.contains_key("AMD Radeon RX 7900 XTX"));
        assert!(amd_cards_present_in(&root));
    }

    #[test]
    fn survives_a_card_without_hwmon() {
        let root = fixture_root("nohwmon");
        write_amd_card(&root, "card1", "744C", "0000:29:00.0");
        std::fs::remove_dir_all(root.join("card1/device/hwmon")).unwrap();

        let metrics = read_amd_metrics(&root).unwrap();
        let card = metrics.get("AMD Radeon RX 7900 XTX").unwrap();
        assert_eq!(card.temp, 0.0);
        assert_eq!(card.power_consumption, 0.0);
        assert_eq!(card.power_limit, 0);
        assert_eq!(card.load, 87, "non-hwmon metrics still reported");
    }

    #[test]
    fn names_unknown_cards_from_pci_id() {
        let root = fixture_root("unknown");
        write_amd_card(&root, "card1", "1234", "0000:29:00.0");

        let metrics = read_amd_metrics(&root).unwrap();
        // No table entry; lspci may or may not resolve it on the test host.
        assert_eq!(metrics.len(), 1);
        let name = metrics.keys().next().unwrap();
        assert!(
            name.contains("1234") || name.contains("Radeon") || name.contains("AMD"),
            "unexpected name: {name}"
        );
    }

    #[test]
    fn numbered_suffix_for_identical_cards() {
        let root = fixture_root("twins");
        write_amd_card(&root, "card1", "744C", "0000:29:00.0");
        write_amd_card(&root, "card2", "744C", "0000:2a:00.0");

        let metrics = read_amd_metrics(&root).unwrap();
        assert_eq!(metrics.len(), 2);
        assert!(metrics.contains_key("AMD Radeon RX 7900 XTX"));
        assert!(metrics.contains_key("AMD Radeon RX 7900 XTX #2"));
    }

    #[test]
    fn missing_drm_root_is_empty_not_an_error() {
        let metrics = read_amd_metrics(Path::new("/nonexistent/drm")).unwrap();
        assert!(metrics.is_empty());
        assert!(!amd_cards_present_in(Path::new("/nonexistent/drm")));
    }

    #[test]
    fn parses_active_dpm_state() {
        assert_eq!(parse_active_clock("0: 2371Mhz *\n1: 500Mhz \n"), 2371);
        assert_eq!(parse_active_clock("S: 0Mhz *\n"), 0);
        assert_eq!(parse_active_clock("0: 96Mhz \n1: 456Mhz *\n"), 456);
        assert_eq!(parse_active_clock("0: 1200MHz *\n"), 1200);
        // No active marker at all — nothing to report rather than a wrong clock.
        assert_eq!(parse_active_clock("0: 96Mhz \n1: 456Mhz \n"), 0);
        assert_eq!(parse_active_clock(""), 0);
    }

    #[test]
    fn parses_lspci_mm_fields() {
        let line = "29:00.0 \"VGA compatible controller\" \"Advanced Micro Devices, Inc. [AMD/ATI]\" \"Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]\" -rc8 -p00 \"Sapphire Technology Limited\" \"PULSE RX 7900 XTX\"";
        let fields = parse_quoted_fields(line);
        assert_eq!(fields[0], "29:00.0");
        assert_eq!(
            fields[3],
            "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M]"
        );
        assert_eq!(fields[7], "PULSE RX 7900 XTX");
    }
}
