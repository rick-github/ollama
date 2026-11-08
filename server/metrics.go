package server

import (
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// metricCollectors holds all the prometheus collectors used by the metrics
// system. It is created once in InitMetrics and registered with a private
// prometheus.Registry so that only ollama's metrics are exposed on /metrics.
type metricCollectors struct {
	// Scheduler state gauges (updated on scrape via CollectFn).
	requestsQueued  prometheus.Gauge
	queueCapacity   prometheus.Gauge
	modelsLoaded    prometheus.Gauge
	maxLoadedModels prometheus.Gauge
	numParallel     prometheus.Gauge
	uptimeSeconds   prometheus.Gauge
	buildInfo       *prometheus.GaugeVec
	gpuMemoryTotal  *prometheus.GaugeVec
	gpuMemoryFree   *prometheus.GaugeVec
	sysMemTotal     prometheus.Gauge
	sysMemFree      prometheus.Gauge
	sysSwapFree     prometheus.Gauge
	contextLength   *prometheus.GaugeVec
	peakMemoryBytes *prometheus.GaugeVec
	tokensPerSecond *prometheus.GaugeVec

	// HTTP request counter.
	httpRequestsTotal *prometheus.CounterVec

	// Per-model duration counters (accumulated as counter of seconds).
	totalDuration      *prometheus.CounterVec
	loadDuration       *prometheus.CounterVec
	promptEvalDuration *prometheus.CounterVec
	evalDuration       *prometheus.CounterVec

	// Per-model token count counters.
	promptEvalCount  *prometheus.CounterVec
	promptEvalCached *prometheus.CounterVec
	evalCount        *prometheus.CounterVec

	// Model lifecycle counters.
	modelLoadTotal   *prometheus.CounterVec
	modelUnloadTotal *prometheus.CounterVec

	// Time to first token / first byte.
	timeToFirstToken *prometheus.HistogramVec

	// Request error and completion counters.
	requestErrors     *prometheus.CounterVec
	requestsCompleted *prometheus.CounterVec

	// Cloud request counter.
	cloudRequests *prometheus.CounterVec

	// Cloud usage gauges (from /api/usage with group_by=model).
	cloudUsageTotalCost         *prometheus.GaugeVec
	cloudUsageTotalRequests     *prometheus.GaugeVec
	cloudUsageRequests          *prometheus.GaugeVec
	cloudUsageInputTokens       *prometheus.GaugeVec
	cloudUsageCachedInputTokens *prometheus.GaugeVec
	cloudUsageOutputTokens      *prometheus.GaugeVec
	cloudUsageCost              *prometheus.GaugeVec
	cloudCostPerToken           *prometheus.GaugeVec
	cloudCostPerPromptToken     *prometheus.GaugeVec
	cloudCostPerEvalToken       *prometheus.GaugeVec

	// Cloud token count gauges (cumulative per model).
	cloudPromptTokens       *prometheus.GaugeVec
	cloudCachedPromptTokens *prometheus.GaugeVec
	cloudEvalTokens         *prometheus.GaugeVec

	// Cloud build info (set to 1 per model/commit/build_time combination).
	cloudBuildInfo *prometheus.GaugeVec

	// Cloud balance gauges (from /api/balance).
	cloudIncludedBalance   prometheus.Gauge
	cloudIncludedAllowance prometheus.Gauge
	cloudPurchasedBalance  prometheus.Gauge
	cloudSessionRemaining  prometheus.Gauge
	cloudWeeklyRemaining   prometheus.Gauge

	// Per-model cost derived from balance deltas (non-legacy plans).
	cloudBalanceCost *prometheus.GaugeVec

	// Per-model usage consumption from legacy session/weekly percentage deltas.
	cloudSessionUsageConsumed *prometheus.GaugeVec
	cloudWeeklyUsageConsumed  *prometheus.GaugeVec
}

// modelLabelValues returns the label values for per-model metrics. The auth
// label is always the actual auth token (e.g. "anon" when no Authorization
// header). Global aggregation is left to the scraper.
func modelLabelValues(model, reason, auth string, includeReason bool) []string {
	r := reason
	if !includeReason {
		r = ""
	}
	return []string{model, r, auth}
}

// curriedModelCounter returns a function that increments the counter for the
// given model/reason/auth combination. It handles the variable-label
// (reason present or not) by using the appropriate label set.
func curriedModelCounter(cv *prometheus.CounterVec, model, reason, auth string, includeReason bool) prometheus.Counter {
	labels := modelLabelValues(model, reason, auth, includeReason)
	return cv.WithLabelValues(labels...)
}

// curriedModelGauge returns a gauge for the given model/reason/auth combination.
func curriedModelGauge(gv *prometheus.GaugeVec, model, reason, auth string, includeReason bool) prometheus.Gauge {
	labels := modelLabelValues(model, reason, auth, includeReason)
	return gv.WithLabelValues(labels...)
}

// InitMetrics creates and registers all prometheus collectors with a fresh
// registry. The scheduler is stored so that custom collectors can read
// live scheduler state (queue depth, loaded models, GPU memory) at scrape
// time.
func InitMetrics(sched *Scheduler) *prometheus.Registry {
	reg := prometheus.NewRegistry()

	mc := &metricCollectors{
		requestsQueued: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_requests_queued",
			Help: "Number of requests waiting for a model runner.",
		}),
		queueCapacity: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_queue_capacity",
			Help: "Maximum number of requests that can be queued (OLLAMA_MAX_QUEUE).",
		}),
		modelsLoaded: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_models_loaded",
			Help: "Number of models currently loaded in memory.",
		}),
		maxLoadedModels: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_max_loaded_models",
			Help: "Maximum number of models allowed to be loaded (OLLAMA_MAX_LOADED_MODELS).",
		}),
		numParallel: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_num_parallel",
			Help: "Default number of parallel requests per model (OLLAMA_NUM_PARALLEL).",
		}),
		uptimeSeconds: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_uptime_seconds",
			Help: "Time since the server started in seconds.",
		}),
		buildInfo: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_build_info",
			Help: "Ollama build information.",
		}, []string{"version"}),

		gpuMemoryTotal: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_gpu_memory_total_bytes",
			Help: "Total memory of the GPU in bytes.",
		}, []string{"gpu_id", "library"}),
		gpuMemoryFree: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_gpu_memory_free_bytes",
			Help: "Free memory of the GPU in bytes.",
		}, []string{"gpu_id", "library"}),

		sysMemTotal: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_system_memory_total_bytes",
			Help: "Total system memory in bytes.",
		}),
		sysMemFree: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_system_memory_free_bytes",
			Help: "Free system memory in bytes.",
		}),
		sysSwapFree: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_system_swap_free_bytes",
			Help: "Free system swap in bytes.",
		}),

		contextLength: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_context_length",
			Help: "Context length the model was loaded with.",
		}, []string{"model", "num_parallel"}),

		peakMemoryBytes: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_peak_memory_bytes",
			Help: "The peak memory used during computation in bytes.",
		}, []string{"model", "reason", "auth"}),

		tokensPerSecond: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_tokens_per_second",
			Help: "Generated tokens per second.",
		}, []string{"model", "auth"}),

		httpRequestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "http_requests_total",
			Help: "The total number of requests on the endpoints.",
		}, []string{"action", "status", "status_code", "auth"}),

		totalDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_total_duration_seconds",
			Help: "The request total duration in seconds.",
		}, []string{"model", "reason", "auth"}),
		loadDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_load_duration_seconds",
			Help: "The request load duration in seconds.",
		}, []string{"model", "reason", "auth"}),
		promptEvalDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_prompt_eval_duration_seconds",
			Help: "The prompt evaluation duration in seconds.",
		}, []string{"model", "reason", "auth"}),
		evalDuration: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_eval_duration_seconds",
			Help: "The token evaluation duration in seconds.",
		}, []string{"model", "reason", "auth"}),

		promptEvalCount: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_prompt_eval_total",
			Help: "The number of prompt token evaluated.",
		}, []string{"model", "reason", "auth"}),
		promptEvalCached: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_prompt_eval_cached_total",
			Help: "The number of cached prompt tokens (cache hits).",
		}, []string{"model", "reason", "auth"}),
		evalCount: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_eval_total",
			Help: "The number of token evaluated.",
		}, []string{"model", "reason", "auth"}),

		modelLoadTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_model_load_total",
			Help: "Total number of model load events.",
		}, []string{"model", "auth"}),
		modelUnloadTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_model_unload_total",
			Help: "Total number of model unload events.",
		}, []string{"model", "auth"}),

		timeToFirstToken: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "ollama_time_to_first_token_seconds",
			Help:    "Time from request start to first token generated (local) or first byte received (cloud).",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 15),
		}, []string{"model", "auth"}),

		requestErrors: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_request_errors_total",
			Help: "Total number of request errors.",
		}, []string{"action", "model", "error_type", "auth"}),
		requestsCompleted: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_requests_completed_total",
			Help: "Total number of completed inference requests.",
		}, []string{"action", "model", "auth"}),

		cloudRequests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "ollama_cloud_requests_total",
			Help: "Total number of cloud-proxied inference requests.",
		}, []string{"action", "model", "auth"}),

		cloudUsageTotalCost: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_total_cost",
			Help: "Total cloud usage cost (usage_usd) for the polled range.",
		}, []string{"range"}),

		cloudUsageTotalRequests: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_total_requests",
			Help: "Total cloud request count for the polled range.",
		}, []string{"range"}),

		cloudUsageRequests: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_requests",
			Help: "Cloud-reported per-model request count for the polled range.",
		}, []string{"model", "range"}),

		cloudUsageInputTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_input_tokens",
			Help: "Cloud-reported per-model input tokens for the polled range.",
		}, []string{"model"}),

		cloudUsageCachedInputTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_cached_input_tokens",
			Help: "Cloud-reported per-model cached input tokens for the polled range.",
		}, []string{"model"}),

		cloudUsageOutputTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_output_tokens",
			Help: "Cloud-reported per-model output tokens for the polled range.",
		}, []string{"model"}),

		cloudUsageCost: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_usage_cost",
			Help: "Estimated cumulative cloud usage cost per model for the current session allocation period.",
		}, []string{"model"}),

		cloudCostPerToken: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_cost_per_token",
			Help: "Estimated cloud usage cost per combined token (prompt + eval) for each model.",
		}, []string{"model"}),

		cloudCostPerPromptToken: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_cost_per_prompt_token",
			Help: "Estimated cloud usage cost per prompt token for each model.",
		}, []string{"model"}),

		cloudCostPerEvalToken: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_cost_per_eval_token",
			Help: "Estimated cloud usage cost per eval (output) token for each model.",
		}, []string{"model"}),

		// The following three token gauges overlap with ollama_prompt_eval_total,
		// ollama_prompt_eval_cached_total, and ollama_eval_total (filtered to cloud
		// models).  They exist because usageTracker maintains its own per-model
		// cumulative counts to compute cost-per-token rates (see usageTracker.update
		// in cloud_metrics.go), and emitting them as simple {model}-labeled gauges
		// provides a direct readout of the tracker's state without needing to
		// aggregate across auth/reason labels.
		cloudPromptTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_prompt_tokens",
			Help: "Cumulative cloud prompt tokens per model.",
		}, []string{"model"}),

		cloudCachedPromptTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_cached_prompt_tokens",
			Help: "Cumulative cloud cached prompt tokens per model.",
		}, []string{"model"}),

		cloudEvalTokens: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_eval_tokens",
			Help: "Cumulative cloud eval (output) tokens per model.",
		}, []string{"model"}),

		cloudBuildInfo: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_build_info",
			Help: "Cloud service build information per model (always 1).",
		}, []string{"model", "commit", "build_time"}),

		cloudIncludedBalance: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_cloud_included_balance_usd",
			Help: "Remaining included (plan) credits in USD (non-legacy plans).",
		}),
		cloudIncludedAllowance: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_cloud_included_allowance_usd",
			Help: "Total included credits available for the current period in USD (non-legacy plans).",
		}),
		cloudPurchasedBalance: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_cloud_purchased_balance_usd",
			Help: "Remaining purchased credits in USD.",
		}),
		cloudSessionRemaining: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_cloud_session_remaining_percent",
			Help: "Remaining session usage percentage (legacy plans, 0-100).",
		}),
		cloudWeeklyRemaining: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "ollama_cloud_weekly_remaining_percent",
			Help: "Remaining weekly usage percentage (legacy plans, 0-100).",
		}),

		cloudBalanceCost: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_balance_cost",
			Help: "Per-model cost derived from cloud balance deltas (non-legacy plans).",
		}, []string{"model"}),

		cloudSessionUsageConsumed: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_session_usage_consumed",
			Help: "Per-model session usage consumed in percentage points (legacy plans).",
		}, []string{"model"}),

		cloudWeeklyUsageConsumed: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "ollama_cloud_weekly_usage_consumed",
			Help: "Per-model weekly usage consumed in percentage points (legacy plans).",
		}, []string{"model"}),
	}

	// Set static gauge values.
	mc.maxLoadedModels.Set(float64(envconfig.MaxRunners()))
	mc.numParallel.Set(float64(envconfig.NumParallel()))
	mc.buildInfo.WithLabelValues(version.Version).Set(1)

	// Register a custom collector that updates dynamic scheduler state
	// gauges at scrape time. This avoids needing to continuously update them.
	schedulerCollector := newSchedulerCollector(sched, mc)
	reg.MustRegister(schedulerCollector)

	// Register all the counter and gauge vecs.
	reg.MustRegister(
		mc.httpRequestsTotal,
		mc.totalDuration,
		mc.loadDuration,
		mc.promptEvalDuration,
		mc.evalDuration,
		mc.promptEvalCount,
		mc.promptEvalCached,
		mc.evalCount,
		mc.peakMemoryBytes,
		mc.tokensPerSecond,
		mc.modelLoadTotal,
		mc.modelUnloadTotal,
		mc.timeToFirstToken,
		mc.requestErrors,
		mc.requestsCompleted,
		mc.cloudRequests,
		mc.cloudUsageTotalCost,
		mc.cloudUsageTotalRequests,
		mc.cloudUsageRequests,
		mc.cloudUsageInputTokens,
		mc.cloudUsageCachedInputTokens,
		mc.cloudUsageOutputTokens,
		mc.cloudUsageCost,
		mc.cloudCostPerToken,
		mc.cloudCostPerPromptToken,
		mc.cloudCostPerEvalToken,
		mc.cloudPromptTokens,
		mc.cloudCachedPromptTokens,
		mc.cloudEvalTokens,
		mc.cloudBuildInfo,
		mc.cloudIncludedBalance,
		mc.cloudIncludedAllowance,
		mc.cloudPurchasedBalance,
		mc.cloudSessionRemaining,
		mc.cloudWeeklyRemaining,
		mc.cloudBalanceCost,
		mc.cloudSessionUsageConsumed,
		mc.cloudWeeklyUsageConsumed,
	)

	sched.metrics = mc
	return reg
}

// schedulerCollector implements prometheus.Collector. It updates the
// dynamic scheduler-state gauges (queue depth, loaded models, GPU memory,
// system memory, context length, uptime, tokens-per-second) at scrape time
// by reading live scheduler state, then delegates to the pre-registered
// gauges.
type schedulerCollector struct {
	sched *Scheduler
	mc    *metricCollectors

	// Dynamic gauges that we update in Describe/Collect. We use unregistered
	// gauges and emit them directly, plus we update the registered GaugeVecs.
	desc *prometheus.Desc
}

func newSchedulerCollector(sched *Scheduler, mc *metricCollectors) *schedulerCollector {
	return &schedulerCollector{
		sched: sched,
		mc:    mc,
		desc:  prometheus.NewDesc("ollama_scheduler_state", "Scheduler state gauges (internal)", nil, nil),
	}
}

func (sc *schedulerCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- sc.desc
}

func (sc *schedulerCollector) Collect(ch chan<- prometheus.Metric) {
	sched := sc.sched

	// Queue depth and loaded models.
	sched.loadedMu.Lock()
	loaded := len(sched.loaded)
	var ctxLengths []contextLengthSnapshot
	for _, r := range sched.loaded {
		if r.Options != nil {
			snap := contextLengthSnapshot{
				model:        r.modelName,
				numCtx:       r.Options.NumCtx,
				numParallel:  r.numParallel,
				numBatch:     r.Options.NumBatch,
				numGPU:       r.Options.NumGPU,
				numThread:    r.Options.NumThread,
				numCtxAuto:   r.numCtxAuto,
				contextShift: r.contextShift,
				trainContext: r.trainContext,
				runner:       r.runner,
				vramSize:     r.vramSize,
				totalSize:    r.totalSize,
			}
			if r.Options.UseMMap != nil {
				if *r.Options.UseMMap {
					snap.useMMap = 1
				} else {
					snap.useMMap = 0
				}
			} else {
				snap.useMMap = -1
			}
			ctxLengths = append(ctxLengths, snap)
		}
	}
	sched.loadedMu.Unlock()

	sc.mc.requestsQueued.Set(float64(len(sched.pendingReqCh)))
	sc.mc.queueCapacity.Set(float64(cap(sched.pendingReqCh)))
	sc.mc.modelsLoaded.Set(float64(loaded))

	// Uptime.
	sched.metricsMu.Lock()
	startTime := sched.startTime
	sched.metricsMu.Unlock()
	sc.mc.uptimeSeconds.Set(time.Since(startTime).Seconds())

	// Emit the static gauges.
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_requests_queued", "Number of requests waiting for a model runner.", nil, nil), prometheus.GaugeValue, float64(len(sched.pendingReqCh)))
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_queue_capacity", "Maximum number of requests that can be queued (OLLAMA_MAX_QUEUE).", nil, nil), prometheus.GaugeValue, float64(cap(sched.pendingReqCh)))
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_models_loaded", "Number of models currently loaded in memory.", nil, nil), prometheus.GaugeValue, float64(loaded))

	// Per-model queue depth.
	queuedByModelDesc := prometheus.NewDesc("ollama_requests_queued_by_model", "Number of queued requests waiting for a model runner, broken out by model.", []string{"model"}, nil)
	sched.pendingMu.Lock()
	for model, count := range sched.pendingModels {
		ch <- prometheus.MustNewConstMetric(queuedByModelDesc, prometheus.GaugeValue, float64(count), model)
	}
	sched.pendingMu.Unlock()
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_max_loaded_models", "Maximum number of models allowed to be loaded (OLLAMA_MAX_LOADED_MODELS).", nil, nil), prometheus.GaugeValue, float64(envconfig.MaxRunners()))
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_num_parallel", "Default number of parallel requests per model (OLLAMA_NUM_PARALLEL).", nil, nil), prometheus.GaugeValue, float64(envconfig.NumParallel()))
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_uptime_seconds", "Time since the server started in seconds.", nil, nil), prometheus.GaugeValue, time.Since(startTime).Seconds())
	ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_build_info", "Ollama build information.", []string{"version"}, nil), prometheus.GaugeValue, 1, version.Version)

	// GPU memory.
	sched.cachedGpusMu.Lock()
	cachedGpus := sched.cachedGpus
	sched.cachedGpusMu.Unlock()

	gpuTotalDesc := prometheus.NewDesc("ollama_gpu_memory_total_bytes", "Total memory of the GPU in bytes.", []string{"gpu_id", "library"}, nil)
	gpuFreeDesc := prometheus.NewDesc("ollama_gpu_memory_free_bytes", "Free memory of the GPU in bytes.", []string{"gpu_id", "library"}, nil)
	for _, g := range cachedGpus {
		ch <- prometheus.MustNewConstMetric(gpuTotalDesc, prometheus.GaugeValue, float64(g.TotalMemory), g.ID, g.Library)
		ch <- prometheus.MustNewConstMetric(gpuFreeDesc, prometheus.GaugeValue, float64(g.FreeMemory), g.ID, g.Library)
	}

	// System memory.
	sched.cachedSysMemMu.Lock()
	sysMem := sched.cachedSysMem
	sched.cachedSysMemMu.Unlock()

	if sysMem.totalMemory > 0 {
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_system_memory_total_bytes", "Total system memory in bytes.", nil, nil), prometheus.GaugeValue, float64(sysMem.totalMemory))
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_system_memory_free_bytes", "Free system memory in bytes.", nil, nil), prometheus.GaugeValue, float64(sysMem.freeMemory))
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_system_swap_free_bytes", "Free system swap in bytes.", nil, nil), prometheus.GaugeValue, float64(sysMem.freeSwap))
	}

	// Context length per loaded model.
	ctxLenDesc := prometheus.NewDesc("ollama_context_length", "Context length the model was loaded with.", []string{"model", "num_parallel"}, nil)
	for _, cl := range ctxLengths {
		ch <- prometheus.MustNewConstMetric(ctxLenDesc, prometheus.GaugeValue, float64(cl.numCtx), cl.model, strconv.Itoa(cl.numParallel))

		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_num_batch", "Batch size the model was loaded with.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.numBatch), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_num_gpu", "Number of GPU layers the model was loaded with.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.numGPU), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_num_thread", "Number of threads the model was loaded with.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.numThread), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_train_context", "Model training context length.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.trainContext), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_use_mmap", "Whether mmap is enabled (1=true, 0=false, -1=auto/unset).", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.useMMap), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_num_ctx_auto", "Whether context size was auto-selected (1/0).", []string{"model"}, nil), prometheus.GaugeValue, boolToFloat(cl.numCtxAuto), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_context_shift", "Whether context shift is enabled (1/0).", []string{"model"}, nil), prometheus.GaugeValue, boolToFloat(cl.contextShift), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_vram_size_bytes", "VRAM size of the loaded model in bytes.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.vramSize), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_total_size_bytes", "Total size of the loaded model in bytes.", []string{"model"}, nil), prometheus.GaugeValue, float64(cl.totalSize), cl.model)
		ch <- prometheus.MustNewConstMetric(prometheus.NewDesc("ollama_runner_info", "Runner backend information (always 1).", []string{"model", "runner"}, nil), prometheus.GaugeValue, 1, cl.model, cl.runner)
	}

	// Tokens per second (computed from eval count and eval duration counters).
	sc.emitTokensPerSecond(ch)

	// Peak memory (read from the scheduler's peak memory map).
	sched.metricsMu.Lock()
	for key, peakBytes := range sched.modelPeakMemoryBytes {
		labels := peakMemoryLabels(key)
		ch <- prometheus.MustNewConstMetric(
			prometheus.NewDesc("ollama_peak_memory_bytes", "The peak memory used during computation in bytes.", peakMemoryLabelNames(), nil),
			prometheus.GaugeValue, float64(peakBytes), labels...)
	}
	sched.metricsMu.Unlock()
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

// peakMemoryLabelNames returns the label names for the peak memory metric.
func peakMemoryLabelNames() []string {
	return []string{"model", "reason", "auth"}
}

// peakMemoryLabels converts a modelMetricKey to label values for peak memory.
func peakMemoryLabels(key modelMetricKey) []string {
	reason := key.reason
	if !key.reasonSet {
		reason = ""
	}
	return []string{key.model, reason, key.auth}
}

// emitTokensPerSecond computes tokens-per-second from the eval count and
// eval duration counters and emits them as const metrics.
func (sc *schedulerCollector) emitTokensPerSecond(ch chan<- prometheus.Metric) {
	// We need to read the counter values. Since we use prometheus
	// counters, we can't easily read them back. Instead, we track
	// the eval count and eval duration in the scheduler's maps (which
	// we still maintain for this purpose) and compute TPS from those.
	sc.sched.metricsMu.Lock()
	defer sc.sched.metricsMu.Unlock()

	type tpsKey struct {
		auth  string
		model string
	}
	tpsEvalCount := map[tpsKey]uint64{}
	tpsEvalDuration := map[tpsKey]uint64{}
	for key, count := range sc.sched.modelEvalCount {
		k := tpsKey{auth: key.auth, model: key.model}
		tpsEvalCount[k] += count
	}
	for key, dur := range sc.sched.modelEvalDuration {
		k := tpsKey{auth: key.auth, model: key.model}
		tpsEvalDuration[k] += dur
	}

	desc := prometheus.NewDesc("ollama_tokens_per_second", "Generated tokens per second.", []string{"model", "auth"}, nil)
	for k, count := range tpsEvalCount {
		durNs := tpsEvalDuration[k]
		var tps float64
		if durNs > 0 {
			tps = float64(count) / (float64(durNs) / float64(time.Second))
		}
		ch <- prometheus.MustNewConstMetric(desc, prometheus.GaugeValue, tps, k.model, k.auth)
	}
}

// MetricsHandler serves the Prometheus metrics endpoint.
func (s *Server) MetricsHandler(c *gin.Context) {
	handler := promhttp.HandlerFor(s.metricsRegistry, promhttp.HandlerOpts{
		Registry: s.metricsRegistry,
	})
	handler.ServeHTTP(c.Writer, c.Request)
}
