//! Identity of the model a llama-server instance has loaded.
//!
//! llama.cpp exposes this in two places that complement each other: `/props`
//! reports the alias, the quantisation, the active context size, the slot count
//! and the build, while `/v1/models` carries the GGUF tensor metadata
//! (`n_params`, file `size`, `n_ctx_train`, vocabulary). Neither is in
//! `/metrics`, which is why the dashboard previously showed nothing about the
//! model itself.
//!
//! Unlike the inference counters this information only changes when the server
//! is restarted with a different model, so the poller refreshes it on a slow
//! cadence instead of on every tick.

use serde_json::Value;

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ModelInfo {
    /// `model_alias`, or the model id from `/v1/models` when no alias is set.
    pub name: String,
    pub path: String,
    /// Quantisation as reported by llama.cpp, e.g. `Q8_0` or `UD-Q4_K_XL`.
    pub quant: String,
    /// Total parameter count from the GGUF metadata.
    pub params: u64,
    /// Size of the model file on disk.
    pub size_bytes: u64,
    /// Context size the server is actually running with.
    pub n_ctx: u64,
    /// Context size the model was trained for.
    pub n_ctx_train: u64,
    pub slots: u32,
    pub build: String,
    pub vision: bool,
    pub n_vocab: u64,
    pub n_embd: u64,
}

impl ModelInfo {
    /// Whether a model has been identified yet.
    pub fn is_loaded(&self) -> bool {
        !self.name.is_empty()
    }

    /// Apply the subset of fields that `/props` reports.
    pub fn apply_props(&mut self, props: &Value) {
        let name = get_str(props, "model_alias");
        if !name.is_empty() {
            self.name = name;
        }
        let path = get_str(props, "model_path");
        if !path.is_empty() {
            self.path = path;
        }
        let quant = get_str(props, "model_ftype");
        if !quant.is_empty() {
            self.quant = quant;
        }
        let build = get_str(props, "build_info");
        if !build.is_empty() {
            self.build = build;
        }
        if let Some(slots) = props.get("total_slots").and_then(Value::as_u64) {
            self.slots = slots as u32;
        }
        if let Some(vision) = props
            .get("modalities")
            .and_then(|m| m.get("vision"))
            .and_then(Value::as_bool)
        {
            self.vision = vision;
        }
        // Current builds nest the context size under default_generation_settings;
        // older ones report it at the top level.
        let n_ctx = props
            .get("default_generation_settings")
            .and_then(|s| s.get("n_ctx"))
            .or_else(|| props.get("n_ctx"))
            .and_then(Value::as_u64);
        if let Some(n) = n_ctx {
            self.n_ctx = n;
        }
    }

    /// Apply the subset of fields that `/v1/models` reports. Only used to fill
    /// gaps: `/props` is authoritative for anything both endpoints expose.
    pub fn apply_models(&mut self, body: &Value) {
        let Some(model) = body
            .get("data")
            .and_then(Value::as_array)
            .and_then(|models| models.first())
        else {
            return;
        };
        if self.name.is_empty() {
            let id = get_str(model, "id");
            if !id.is_empty() {
                self.name = id;
            }
        }
        let Some(meta) = model.get("meta") else {
            return;
        };
        set_if_present(&mut self.params, get_u64(meta, "n_params"));
        set_if_present(&mut self.size_bytes, get_u64(meta, "size"));
        set_if_present(&mut self.n_ctx_train, get_u64(meta, "n_ctx_train"));
        set_if_present(&mut self.n_vocab, get_u64(meta, "n_vocab"));
        set_if_present(&mut self.n_embd, get_u64(meta, "n_embd"));
        if self.n_ctx == 0 {
            self.n_ctx = get_u64(meta, "n_ctx");
        }
        if self.quant.is_empty() {
            self.quant = get_str(meta, "ftype");
        }
    }
}

fn get_str(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn get_u64(v: &Value, key: &str) -> u64 {
    v.get(key).and_then(Value::as_u64).unwrap_or(0)
}

/// Absent metadata is reported as 0; keep whatever an earlier endpoint supplied
/// instead of blanking it out.
fn set_if_present(target: &mut u64, value: u64) {
    if value > 0 {
        *target = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props() -> Value {
        serde_json::json!({
            "default_generation_settings": { "n_ctx": 262144, "params": { "temperature": 1.0 } },
            "total_slots": 1,
            "model_alias": "qwen3.8-27b-q8",
            "model_ftype": "Q8_0",
            "model_path": "/home/william/models/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf",
            "modalities": { "vision": false, "video": false, "audio": false },
            "chat_template": "{%- set image_count = namespace(value=0) %}",
            "build_info": "b1-3057bb6"
        })
    }

    fn models() -> Value {
        serde_json::json!({
            "object": "list",
            "data": [{
                "id": "qwen3.8-27b-q8",
                "object": "model",
                "owned_by": "llamacpp",
                "meta": {
                    "n_vocab": 248320,
                    "n_ctx": 262144,
                    "n_ctx_train": 262144,
                    "n_embd": 5120,
                    "n_params": 27320697856u64,
                    "size": 29036089344u64,
                    "ftype": "Q8_0"
                }
            }]
        })
    }

    #[test]
    fn reads_props() {
        let mut info = ModelInfo::default();
        info.apply_props(&props());
        assert_eq!(info.name, "qwen3.8-27b-q8");
        assert_eq!(info.quant, "Q8_0");
        assert_eq!(
            info.path,
            "/home/william/models/qwen3.8-27b/Qwen3.8-27B-Q8_0.gguf"
        );
        assert_eq!(info.n_ctx, 262144);
        assert_eq!(info.slots, 1);
        assert_eq!(info.build, "b1-3057bb6");
        assert!(!info.vision);
        assert!(info.is_loaded());
    }

    #[test]
    fn reads_models_metadata() {
        let mut info = ModelInfo::default();
        info.apply_models(&models());
        assert_eq!(info.name, "qwen3.8-27b-q8");
        assert_eq!(info.params, 27_320_697_856);
        assert_eq!(info.size_bytes, 29_036_089_344);
        assert_eq!(info.n_ctx_train, 262144);
        assert_eq!(info.n_vocab, 248320);
        assert_eq!(info.n_embd, 5120);
        assert_eq!(info.quant, "Q8_0");
    }

    #[test]
    fn props_win_over_models_for_shared_fields() {
        let mut info = ModelInfo::default();
        info.apply_props(&props());
        // A stale /v1/models payload must not overwrite the running context.
        let stale = serde_json::json!({
            "data": [{ "id": "other", "meta": { "n_ctx": 4096, "ftype": "Q4_0" } }]
        });
        info.apply_models(&stale);
        assert_eq!(info.name, "qwen3.8-27b-q8");
        assert_eq!(info.n_ctx, 262144);
        assert_eq!(info.quant, "Q8_0");
    }

    #[test]
    fn tolerates_missing_and_odd_payloads() {
        let mut info = ModelInfo::default();
        info.apply_props(&serde_json::json!({}));
        info.apply_models(&serde_json::json!({ "data": [] }));
        info.apply_models(&serde_json::json!("not an object"));
        info.apply_models(&serde_json::json!({ "data": [{ "id": "no-meta" }] }));
        assert_eq!(info.name, "no-meta");
        assert_eq!(info.n_ctx, 0);
        assert_eq!(info.params, 0);
    }

    #[test]
    fn top_level_context_is_accepted_for_older_builds() {
        let mut info = ModelInfo::default();
        info.apply_props(&serde_json::json!({ "model_alias": "x", "n_ctx": 8192 }));
        assert_eq!(info.n_ctx, 8192);
    }
}
