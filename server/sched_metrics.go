package server

import (
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// schedulerMetricsState holds the metrics-related fields of the Scheduler.
// It is embedded in Scheduler so that the metrics fields are promoted
// and accessible directly (e.g. s.metrics, s.cachedGpus) without
// changing call sites. Defining it here keeps the Scheduler struct
// definition in sched.go free of metrics-specific fields, reducing
// merge conflicts.
type schedulerMetricsState struct {
	// cachedGpus holds the most recently discovered GPU list, updated when
	// the scheduler probes GPUs in processPending.  Used by the metrics
	// collector to expose per-GPU memory gauges without a fresh discovery
	// round-trip.
	cachedGpus   []ml.DeviceInfo
	cachedGpusMu sync.Mutex

	// cachedSysMem holds the most recently discovered system memory info,
	// updated when the scheduler probes system info in processPending.
	cachedSysMem   systemMemorySnapshot
	cachedSysMemMu sync.Mutex

	// metrics holds the prometheus collectors. Set by InitMetrics.
	metrics *metricCollectors

	// metricsMu protects the peak-memory and token-count maps that are
	// read by the custom collector at scrape time. All counter recording
	// goes directly through the prometheus library (thread-safe), but we
	// keep these maps for derived metrics (tokens-per-second) and for
	// peak-memory gauges that need max-of semantics.
	metricsMu            sync.Mutex
	modelPeakMemoryBytes map[modelMetricKey]uint64
	modelEvalCount       map[modelMetricKey]uint64
	modelEvalDuration    map[modelMetricKey]uint64
	startTime            time.Time
}

type schedulerMetrics struct {
	requestsQueued  int
	queueCapacity   int
	modelsLoaded    int
	maxLoadedModels uint
	numParallel     uint
	gpuMemory       []gpuMemorySnapshot
	contextLengths  []contextLengthSnapshot
	systemMemory    systemMemorySnapshot
}

type gpuMemorySnapshot struct {
	id          string
	library     string
	totalMemory uint64
	freeMemory  uint64
}

type systemMemorySnapshot struct {
	totalMemory uint64
	freeMemory  uint64
	freeSwap    uint64
}

type contextLengthSnapshot struct {
	model        string
	numCtx       int
	numParallel  int
	numBatch     int
	numGPU       int
	useMMap      int // 1 (true), 0 (false), -1 (nil/auto)
	numThread    int
	numCtxAuto   bool
	contextShift bool
	trainContext int
	runner       string
	vramSize     uint64
	totalSize    uint64
}

// noAuthLabel is retained as a sentinel for the auth field in internal
// bookkeeping maps (modelMetricKey.auth). It is no longer written to
// prometheus counters — global aggregation is left to the scraper via
// `sum by (model, reason) (...)` queries.
const noAuthLabel = "\x00"

type modelMetricKey struct {
	model     string
	reasonSet bool
	reason    string
	auth      string
}

// collectMetrics returns a snapshot of scheduler state for the custom
// prometheus collector. The pending queue is a buffered channel so its
// length is a race-safe read; loaded is guarded by loadedMu.
func (s *Scheduler) collectMetrics() schedulerMetrics {
	s.loadedMu.Lock()
	loaded := len(s.loaded)
	var ctxLengths []contextLengthSnapshot
	for _, r := range s.loaded {
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
	s.loadedMu.Unlock()

	s.cachedGpusMu.Lock()
	cachedGpus := s.cachedGpus
	s.cachedGpusMu.Unlock()

	var gpus []gpuMemorySnapshot
	for _, g := range cachedGpus {
		gpus = append(gpus, gpuMemorySnapshot{
			id:          g.ID,
			library:     g.Library,
			totalMemory: g.TotalMemory,
			freeMemory:  g.FreeMemory,
		})
	}

	s.cachedSysMemMu.Lock()
	sysMem := s.cachedSysMem
	s.cachedSysMemMu.Unlock()

	return schedulerMetrics{
		requestsQueued:  len(s.pendingReqCh),
		queueCapacity:   cap(s.pendingReqCh),
		modelsLoaded:    loaded,
		maxLoadedModels: envconfig.MaxRunners(),
		numParallel:     uint(envconfig.NumParallel()),
		gpuMemory:       gpus,
		contextLengths:  ctxLengths,
		systemMemory:    sysMem,
	}
}

func (s *Scheduler) recordHTTPRequests(action string, statusCode int, status string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	statusCodeStr := strconv.Itoa(statusCode)
	for _, a := range []string{"all", action} {
		s.metrics.httpRequestsTotal.WithLabelValues(a, status, statusCodeStr, auth).Inc()
	}
}

func (s *Scheduler) recordPromptAndEvalMetrics(model string, reason string, includeReason bool, totalDuration, loadDuration, promptEvalDuration, evalDuration time.Duration, promptEvalCount, evalCount int, auth string) {
	s.recordPromptAndEvalMetricsWithMemory(
		model,
		reason,
		includeReason,
		totalDuration,
		loadDuration,
		promptEvalDuration,
		evalDuration,
		promptEvalCount,
		evalCount,
		0,
		auth,
		0,
	)
}

func (s *Scheduler) recordPromptAndEvalMetricsWithMemory(model string, reason string, includeReason bool, totalDuration, loadDuration, promptEvalDuration, evalDuration time.Duration, promptEvalCount, evalCount int, peakMemoryBytes uint64, auth string, promptEvalCachedCount int) {
	if s == nil || s.metrics == nil {
		return
	}

	// Record per-user entries via prometheus counters. Global aggregation
	// is left to the scraper (e.g. `sum by (model, reason) (...)`).
	c := curriedModelCounter
	c(s.metrics.totalDuration, model, reason, auth, includeReason).Add(totalDuration.Seconds())
	c(s.metrics.loadDuration, model, reason, auth, includeReason).Add(loadDuration.Seconds())
	c(s.metrics.promptEvalDuration, model, reason, auth, includeReason).Add(promptEvalDuration.Seconds())
	c(s.metrics.evalDuration, model, reason, auth, includeReason).Add(evalDuration.Seconds())
	c(s.metrics.promptEvalCount, model, reason, auth, includeReason).Add(float64(promptEvalCount))
	c(s.metrics.promptEvalCached, model, reason, auth, includeReason).Add(float64(promptEvalCachedCount))
	c(s.metrics.evalCount, model, reason, auth, includeReason).Add(float64(evalCount))

	// Track eval count/duration in maps for tokens-per-second computation,
	// and peak memory as a max gauge.
	s.metricsMu.Lock()
	defer s.metricsMu.Unlock()
	if s.modelEvalCount == nil {
		s.modelEvalCount = make(map[modelMetricKey]uint64)
	}
	if s.modelEvalDuration == nil {
		s.modelEvalDuration = make(map[modelMetricKey]uint64)
	}
	if s.modelPeakMemoryBytes == nil {
		s.modelPeakMemoryBytes = make(map[modelMetricKey]uint64)
	}
	key := modelMetricKey{model: model, reasonSet: includeReason, reason: reason, auth: auth}
	s.modelEvalCount[key] += uint64(evalCount)
	s.modelEvalDuration[key] += uint64(evalDuration.Nanoseconds())
	if peakMemoryBytes > 0 && s.modelPeakMemoryBytes[key] < peakMemoryBytes {
		s.modelPeakMemoryBytes[key] = peakMemoryBytes
	}
}

func (s *Scheduler) recordModelLoad(model string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.modelLoadTotal.WithLabelValues(model, auth).Inc()
}

func (s *Scheduler) recordModelUnload(model string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.modelUnloadTotal.WithLabelValues(model, auth).Inc()
}

func (s *Scheduler) recordRequestError(action string, model string, errorType string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.requestErrors.WithLabelValues(action, model, errorType, auth).Inc()
}

func (s *Scheduler) recordRequestCompleted(action string, model string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.requestsCompleted.WithLabelValues(action, model, auth).Inc()
}

func (s *Scheduler) recordCloudRequest(action string, model string, auth string) {
	if s == nil || s.metrics == nil {
		return
	}
	s.metrics.cloudRequests.WithLabelValues(action, model, auth).Inc()
}

func (s *Scheduler) recordCloudBuildInfo(model, commit, buildTime string) {
	if s == nil || s.metrics == nil || commit == "" {
		return
	}
	s.metrics.cloudBuildInfo.WithLabelValues(model, commit, buildTime).Set(1)
}

func (s *Scheduler) recordTTFT(model, auth string, ttft time.Duration) {
	if s == nil || s.metrics == nil || ttft <= 0 {
		return
	}
	s.metrics.timeToFirstToken.WithLabelValues(model, auth).Observe(ttft.Seconds())
}

func routeToAction(route string) string {
	switch route {
	case "/api/chat", "/v1/chat/completions", "/v1/messages":
		return "chat"
	case "/api/embed", "/v1/embeddings":
		return "embed"
	case "/v1/completions":
		return "completions"
	case "/v1/responses":
		return "responses"
	case "/v1/audio/transcriptions":
		return "transcriptions"
	default:
		parts := strings.Split(route, "/")
		if len(parts) > 2 {
			return parts[len(parts)-1]
		}
		return route
	}
}

// metricsAuthToken extracts the bearer token from the Authorization header
// for use as a per-user label in metrics. Returns "anon" when no token is
// present or when OLLAMA_METRICS_NO_AUTH is set. The token is truncated to
// a short prefix so the raw token is not fully exposed in metric labels.
func metricsAuthToken(c *gin.Context) string {
	if envconfig.MetricsDisableAuth() {
		return "anon"
	}
	h := strings.TrimSpace(c.GetHeader("Authorization"))
	if h == "" {
		return "anon"
	}
	var token string
	if len(h) >= 6 && strings.EqualFold(h[:6], "Bearer") {
		token = strings.TrimSpace(h[6:])
		if token == "" {
			return "anon"
		}
	} else {
		token = h
	}
	if len(token) > 8 {
		return token[:8]
	}
	return token
}

func promptEvalCachedCount(p *int) int {
	if p != nil {
		return *p
	}
	return 0
}

func (s *Server) metricsMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		c.Next()

		route := c.FullPath()
		if route == "" {
			return
		}

		responseStatus := c.Writer.Status()
		if s.sched != nil {
			auth := metricsAuthToken(c)
			action := routeToAction(route)
			s.sched.recordHTTPRequests(action, responseStatus, http.StatusText(responseStatus), auth)

			// Record completed inference requests.
			switch action {
			case "chat", "generate", "embed", "embeddings":
				model, _ := c.Get("model")
				modelName, _ := model.(string)
				if modelName == "" {
					if cloudModel, ok := c.Get("cloudModel"); ok {
						modelName, _ = cloudModel.(string)
					}
				}
				s.sched.recordRequestCompleted(action, strings.ToLower(modelName), auth)
			}

			// Record errors for non-successful status codes.
			if responseStatus >= 400 {
				model, _ := c.Get("model")
				modelName, _ := model.(string)
				s.sched.recordRequestError(action, strings.ToLower(modelName), http.StatusText(responseStatus), auth)
			}
		}
	}
}
