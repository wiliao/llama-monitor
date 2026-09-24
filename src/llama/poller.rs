use std::time::{Duration, Instant};

use crate::state::AppState;

use super::metrics::parse_prometheus_metrics;

const LLAMA_POLL_INTERVAL: Duration = Duration::from_secs(1);

pub async fn llama_metrics_poller(state: AppState) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()
        .unwrap();

    // Cumulative counters from the previous poll. Kept across iterations so the
    // completion-time counter bump can be turned into a per-request rate.
    let mut previous_counters: Option<(f64, f64, f64, f64)> = None;
    // Previous /slots reading of the in-flight task: (task, prompt processed,
    // decoded, when) — used to derive live per-second rates.
    let mut last_slot_sample: Option<(u64, u64, u64, Instant)> = None;

    loop {
        // Determine port: from config if started via UI, else default 8080
        let port = state
            .server_config
            .lock()
            .unwrap()
            .as_ref()
            .map(|c| c.port)
            .unwrap_or(8080);

        let base = format!("http://127.0.0.1:{port}");

        // Poll /health first to detect if any server is reachable
        let server_reachable = if let Ok(resp) = client.get(format!("{base}/health")).send().await {
            if let Ok(body) = resp.text().await {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
                    let mut m = state.llama_metrics.lock().unwrap();
                    m.status = json
                        .get("status")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if !server_reachable {
            {
                let mut m = state.llama_metrics.lock().unwrap();
                *m = super::metrics::LlamaMetrics::default();
            }
            // Counters restart with the server; a stale baseline would make the
            // first deltas negative and skip a request.
            previous_counters = None;
            last_slot_sample = None;
            tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
            continue;
        }

        // Poll /metrics
        if let Ok(resp) = client.get(format!("{base}/metrics")).send().await
            && let Ok(body) = resp.text().await
        {
            let prom = parse_prometheus_metrics(&body);

            // llama.cpp only bumps these cumulative counters when a task *finishes*
            // (verified: identical across polls while a request is in flight), so a
            // delta between two polls is exactly one request's tokens/seconds. This
            // build reports the per-request gauges as 0, which would otherwise leave
            // the panel showing the lifetime average — a number that never moves.
            let counters = (
                prom.prompt_tokens_total,
                prom.prompt_seconds_total,
                prom.predicted_tokens_total,
                prom.predicted_seconds_total,
            );
            let mut prompt_tps = prom.prompt_tokens_per_sec;
            let mut gen_tps = prom.predicted_tokens_per_sec;
            match previous_counters {
                Some((p_tok, p_sec, g_tok, g_sec)) => {
                    let (d_tok, d_sec) = (counters.0 - p_tok, counters.1 - p_sec);
                    if d_tok > 0.0 && d_sec > 0.0 {
                        prompt_tps = d_tok / d_sec;
                    }
                    let (d_tok, d_sec) = (counters.2 - g_tok, counters.3 - g_sec);
                    if d_tok > 0.0 && d_sec > 0.0 {
                        gen_tps = d_tok / d_sec;
                    }
                }
                None => {
                    // First pass (or just after a server restart): seed with the
                    // lifetime average so the panel is not blank until it completes
                    // a request.
                    if prompt_tps <= 0.0 && counters.1 > 0.0 {
                        prompt_tps = counters.0 / counters.1;
                    }
                    if gen_tps <= 0.0 && counters.3 > 0.0 {
                        gen_tps = counters.2 / counters.3;
                    }
                }
            }
            previous_counters = Some(counters);

            let mut m = state.llama_metrics.lock().unwrap();
            // Only overwrite when a rate is actually known, so the last completed
            // request's speed stays on screen between requests instead of decaying
            // back to the lifetime average.
            if prompt_tps > 0.0 {
                m.prompt_tokens_per_sec = prompt_tps;
            }
            if gen_tps > 0.0 {
                m.generation_tokens_per_sec = gen_tps;
            }
            m.prompt_tokens_total = prom.prompt_tokens_total as u64;
            m.predicted_tokens_total = prom.predicted_tokens_total as u64;
            // NOT kv_cache_tokens: n_tokens_max is a monotonic high-water mark, so
            // it looks frozen. The live context comes from /slots below.
            m.kv_cache_peak = prom.n_tokens_max;
            m.requests_processing = prom.requests_processing;
        }

        // Poll /slots — get per-slot processing state + total context
        if let Ok(resp) = client.get(format!("{base}/slots")).send().await
            && let Ok(body) = resp.text().await
            && let Ok(slots) = serde_json::from_str::<Vec<serde_json::Value>>(&body)
        {
            let mut idle = 0u32;
            let mut processing = 0u32;
            let num_slots = slots.len() as u64;
            let mut per_slot_ctx = 0u64;
            // Current context = the largest prompt a slot holds. Idle slots keep
            // reporting their last task, so this stays meaningful between turns.
            let mut context_tokens = 0u64;
            // Live state of the in-flight slot:
            // (task, prompt total, processed, cached, decoded so far)
            let mut live: Option<(u64, u64, u64, u64, u64)> = None;
            for slot in &slots {
                let field = |name: &str| slot.get(name).and_then(|v| v.as_u64()).unwrap_or(0);
                context_tokens = context_tokens.max(field("n_prompt_tokens"));
                if slot
                    .get("is_processing")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
                {
                    processing += 1;
                    // `next_token` is a LIST of per-sequence decode states whose
                    // n_decoded advances as tokens are produced — the only live
                    // per-token progress llama.cpp exposes. It is empty during
                    // prompt evaluation, before anything has been decoded.
                    let decoded = slot
                        .get("next_token")
                        .and_then(|v| v.as_array())
                        .and_then(|seqs| seqs.first())
                        .and_then(|seq| seq.get("n_decoded"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0);
                    live = Some((
                        field("id_task"),
                        field("n_prompt_tokens"),
                        field("n_prompt_tokens_processed"),
                        field("n_prompt_tokens_cache"),
                        decoded,
                    ));
                } else {
                    idle += 1;
                }
                if let Some(n) = slot.get("n_ctx").and_then(|v| v.as_u64()) {
                    per_slot_ctx = n;
                }
            }

            let (task, prompt_total, prompt_done, prompt_cached, decoded) =
                live.unwrap_or((0, 0, 0, 0, 0));

            // Live rates from /slots deltas. The Prometheus counters only move when
            // a task finishes, so this is what actually animates during a turn.
            let now = Instant::now();
            let mut live_prompt_tps = None;
            let mut live_gen_tps = None;
            if let Some((prev_task, prev_done, prev_decoded, prev_at)) = last_slot_sample {
                let dt = now.duration_since(prev_at).as_secs_f64();
                if dt > 0.05 && task > 0 && task == prev_task {
                    let delta = prompt_done.saturating_sub(prev_done) as f64 / dt;
                    if delta > 0.0 && delta < 10_000.0 {
                        live_prompt_tps = Some(delta);
                    }
                    let delta = decoded.saturating_sub(prev_decoded) as f64 / dt;
                    // A wild delta means a reset or a reused counter, not a rate.
                    if delta > 0.0 && delta < 1000.0 {
                        live_gen_tps = Some(delta);
                    }
                }
            }
            last_slot_sample = Some((task, prompt_done, decoded, now));

            let mut m = state.llama_metrics.lock().unwrap();
            m.slots_idle = idle;
            m.slots_processing = processing;
            m.kv_cache_tokens = context_tokens;
            if per_slot_ctx > 0 {
                m.kv_cache_max = per_slot_ctx * num_slots;
            }
            m.id_task = task;
            m.prompt_tokens = prompt_total;
            m.prompt_tokens_processed = prompt_done;
            m.prompt_tokens_cached = prompt_cached;
            m.gen_tokens = decoded;
            if let Some(tps) = live_prompt_tps {
                m.prompt_tokens_per_sec = tps;
            }
            if let Some(tps) = live_gen_tps {
                m.generation_tokens_per_sec = tps;
            }
        }

        tokio::time::sleep(LLAMA_POLL_INTERVAL).await;
    }
}
