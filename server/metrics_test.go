package server

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

// newTestMetricsServer creates a Server with a real scheduler and metrics
// registry for testing. It returns the server and a gin engine with /metrics
// registered.
func newTestMetricsServer(sched *Scheduler) (*Server, *gin.Engine) {
	if sched == nil {
		sched = &Scheduler{
			pendingReqCh: make(chan *LlmRequest, 512),
			loaded:       map[string]*runnerRef{},
		}
	}
	if sched.modelPeakMemoryBytes == nil {
		sched.modelPeakMemoryBytes = make(map[modelMetricKey]uint64)
	}
	if sched.modelEvalCount == nil {
		sched.modelEvalCount = make(map[modelMetricKey]uint64)
	}
	if sched.modelEvalDuration == nil {
		sched.modelEvalDuration = make(map[modelMetricKey]uint64)
	}
	if sched.startTime.IsZero() {
		sched.startTime = time.Now()
	}
	s := &Server{sched: sched}
	s.metricsRegistry = InitMetrics(sched)
	gin.SetMode(gin.TestMode)
	r := gin.New()
	r.GET("/metrics", s.MetricsHandler)
	return s, r
}

func TestMetricsHandler(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 512),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}
	sched.pendingReqCh <- &LlmRequest{}
	sched.pendingReqCh <- &LlmRequest{}

	_, r := newTestMetricsServer(sched)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	for _, want := range []string{
		"# HELP ollama_requests_queued Number of requests waiting for a model runner.",
		"# TYPE ollama_requests_queued gauge",
		"ollama_requests_queued 2",
		"ollama_queue_capacity 512",
		"ollama_models_loaded 1",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}

func TestMetricsHandlerEmpty(t *testing.T) {
	_, r := newTestMetricsServer(nil)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}
	body := w.Body.String()
	for _, want := range []string{"ollama_requests_queued 0", "ollama_models_loaded 0"} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}

func TestMetricsHandlerBuildInfo(t *testing.T) {
	_, r := newTestMetricsServer(nil)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	if !strings.Contains(body, `# HELP ollama_build_info Ollama build information.`) {
		t.Fatalf("metrics output missing build info metric\n--- got ---\n%s", body)
	}
}

func TestMetricsHandlerSortedHTTPMetrics(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}
	_, r := newTestMetricsServer(sched)

	sched.recordHTTPRequests("chat", http.StatusServiceUnavailable, http.StatusText(http.StatusServiceUnavailable), "anon")
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "anon")
	sched.recordHTTPRequests("embed", http.StatusBadRequest, http.StatusText(http.StatusBadRequest), "anon")
	sched.recordHTTPRequests("all", http.StatusOK, http.StatusText(http.StatusOK), "anon")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	// Only per-user entries are recorded; global aggregation is left to the scraper.
	for _, want := range []string{
		`http_requests_total{action="all",auth="anon",status="Bad Request",status_code="400"} 1`,
		`http_requests_total{action="all",auth="anon",status="OK",status_code="200"} 3`,
		`http_requests_total{action="all",auth="anon",status="Service Unavailable",status_code="503"} 1`,
		`http_requests_total{action="chat",auth="anon",status="OK",status_code="200"} 1`,
		`http_requests_total{action="chat",auth="anon",status="Service Unavailable",status_code="503"} 1`,
		`http_requests_total{action="embed",auth="anon",status="Bad Request",status_code="400"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}

	// No global auth="" entries should be present.
	if strings.Contains(body, `auth=""`) {
		t.Errorf("metrics output should not contain auth=\"\" global aggregate entries\n--- got ---\n%s", body)
	}
}

func TestMetricsHandlerReasonLabels(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{"model-a": {}},
	}
	_, r := newTestMetricsServer(sched)

	sched.recordPromptAndEvalMetricsWithMemory("with-reason", "timeout", true, 2*time.Second, time.Second, 500*time.Millisecond, 500*time.Millisecond, 200, 150, 4096, "token-a", 0)
	sched.recordPromptAndEvalMetricsWithMemory("no-reason", "ignore", false, time.Second, 250*time.Millisecond, 100*time.Millisecond, 150*time.Millisecond, 100, 50, 2048, "token-b", 0)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()

	// Prometheus library uses Go's default float formatting (no trailing zeros).
	if !strings.Contains(body, `ollama_prompt_eval_duration_seconds{auth="token-a",model="with-reason",reason="timeout"} 0.5`) {
		t.Errorf("metrics output missing with-reason sample\n--- got ---\n%s", body)
	}
	// When includeReason=false, reason label is empty string.
	if !strings.Contains(body, `ollama_prompt_eval_duration_seconds{auth="token-b",model="no-reason",reason=""} 0.1`) {
		t.Errorf("metrics output missing no-reason sample\n--- got ---\n%s", body)
	}
	if strings.Contains(body, `ollama_prompt_eval_duration_seconds{auth="token-b",model="no-reason",reason="ignore"}`) {
		t.Fatalf("prompt eval should not include reason value when includeReason=false")
	}

	// Peak memory is emitted as a gauge.
	if !strings.Contains(body, `ollama_peak_memory_bytes{auth="token-b",model="no-reason",reason=""} 2048`) {
		t.Errorf("metrics output missing peak memory sample\n--- got ---\n%s", body)
	}
	if !strings.Contains(body, `ollama_peak_memory_bytes{auth="token-a",model="with-reason",reason="timeout"} 4096`) {
		t.Errorf("metrics output missing reasoned peak memory sample\n--- got ---\n%s", body)
	}
}

func TestMetricsHandlerExpandedMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pendingReqCh = make(chan *LlmRequest, 4)
	sched.loaded = map[string]*runnerRef{"model-a": {}}
	_, r := newTestMetricsServer(sched)

	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "user1")
	sched.recordHTTPRequests("embed", http.StatusBadRequest, http.StatusText(http.StatusBadRequest), "user2")
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true, 1500*time.Millisecond, 500*time.Millisecond, 300*time.Millisecond, 450*time.Millisecond, 131, 62, 120, "user1", 0)
	sched.recordPromptAndEvalMetrics("phi4", "", false, 0, 0, 0, 0, 42, 0, "user2")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	for _, want := range []string{
		`# HELP http_requests_total The total number of requests on the endpoints.`,
		`http_requests_total{action="all",auth="user1",status="OK",status_code="200"} 1`,
		`http_requests_total{action="chat",auth="user1",status="OK",status_code="200"} 1`,
		`http_requests_total{action="embed",auth="user2",status="Bad Request",status_code="400"} 1`,
		`http_requests_total{action="all",auth="user2",status="Bad Request",status_code="400"} 1`,
		`# HELP ollama_build_info Ollama build information.`,
		`ollama_build_info{version="0.0.0"} 1`,
		`# HELP ollama_peak_memory_bytes The peak memory used during computation in bytes.`,
		`ollama_peak_memory_bytes{auth="user1",model="llama3",reason="stop"} 120`,
		"# HELP ollama_total_duration_seconds The request total duration in seconds.",
		`ollama_total_duration_seconds{auth="user1",model="llama3",reason="stop"} 1.5`,
		`ollama_load_duration_seconds{auth="user1",model="llama3",reason="stop"} 0.5`,
		`ollama_prompt_eval_total{auth="user1",model="llama3",reason="stop"} 131`,
		`ollama_prompt_eval_total{auth="user2",model="phi4",reason=""} 42`,
		`ollama_prompt_eval_duration_seconds{auth="user1",model="llama3",reason="stop"} 0.3`,
		`ollama_eval_total{auth="user1",model="llama3",reason="stop"} 62`,
		`ollama_eval_duration_seconds{auth="user1",model="llama3",reason="stop"} 0.45`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}

func TestMetricsHandlerAuthLabels(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{},
	}
	_, r := newTestMetricsServer(sched)

	// Two different auth tokens hitting the same endpoint and model.
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "alice")
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "bob")
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true,
		1500*time.Millisecond, 500*time.Millisecond, 300*time.Millisecond, 450*time.Millisecond,
		131, 62, 120, "alice", 0)
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true,
		800*time.Millisecond, 200*time.Millisecond, 150*time.Millisecond, 300*time.Millisecond,
		80, 40, 90, "bob", 0)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()

	// HTTP request counters should be split per auth token.
	for _, want := range []string{
		`http_requests_total{action="all",auth="alice",status="OK",status_code="200"} 1`,
		`http_requests_total{action="all",auth="bob",status="OK",status_code="200"} 1`,
		`http_requests_total{action="chat",auth="alice",status="OK",status_code="200"} 1`,
		`http_requests_total{action="chat",auth="bob",status="OK",status_code="200"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}

	// Per-model metrics should also be split per auth token.
	for _, want := range []string{
		`ollama_total_duration_seconds{auth="alice",model="llama3",reason="stop"} 1.5`,
		`ollama_total_duration_seconds{auth="bob",model="llama3",reason="stop"} 0.8`,
		`ollama_prompt_eval_total{auth="alice",model="llama3",reason="stop"} 131`,
		`ollama_prompt_eval_total{auth="bob",model="llama3",reason="stop"} 80`,
		`ollama_eval_total{auth="alice",model="llama3",reason="stop"} 62`,
		`ollama_eval_total{auth="bob",model="llama3",reason="stop"} 40`,
		`ollama_peak_memory_bytes{auth="alice",model="llama3",reason="stop"} 120`,
		`ollama_peak_memory_bytes{auth="bob",model="llama3",reason="stop"} 90`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}
}

func TestMetricsAuthToken(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"no header", "", "anon"},
		{"bearer token", "Bearer mytoken", "mytoken"},
		{"bearer lowercase", "bearer mytoken", "mytoken"},
		{"raw token", "mytoken", "mytoken"},
		{"empty bearer", "Bearer", "anon"},
		{"bearer with spaces", "Bearer  multi  space", "multi  s"},
		{"long token truncated", "Bearer verylongtoken123", "verylong"},
		{"raw long token truncated", "verylongtoken123", "verylong"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.header != "" {
				req.Header.Set("Authorization", tt.header)
			}
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = req
			got := metricsAuthToken(c)
			if got != tt.want {
				t.Errorf("metricsAuthToken() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMetricsAuthTokenPrefix(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name   string
		header string
		want   string
	}{
		{"no header still anon", "", "anon"},
		{"short token unchanged", "Bearer abc", "abc"},
		{"exactly 8 chars unchanged", "Bearer 12345678", "12345678"},
		{"long token truncated to 8", "Bearer verylongtoken", "verylong"},
		{"different token different prefix", "Bearer othertoken", "othertok"},
		{"empty bearer still anon", "Bearer", "anon"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.header != "" {
				req.Header.Set("Authorization", tt.header)
			}
			w := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(w)
			c.Request = req
			got := metricsAuthToken(c)
			if got != tt.want {
				t.Errorf("metricsAuthToken() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestMetricsHandlerGlobalAggregate(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{},
	}
	_, r := newTestMetricsServer(sched)

	// Three requests: two from alice, one from bob.
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "alice")
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "alice")
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "bob")
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true,
		1500*time.Millisecond, 500*time.Millisecond, 300*time.Millisecond, 450*time.Millisecond,
		131, 62, 120, "alice", 0)
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true,
		800*time.Millisecond, 200*time.Millisecond, 150*time.Millisecond, 300*time.Millisecond,
		80, 40, 90, "bob", 0)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	// Per-user entries.
	for _, want := range []string{
		`http_requests_total{action="all",auth="alice",status="OK",status_code="200"} 2`,
		`http_requests_total{action="all",auth="bob",status="OK",status_code="200"} 1`,
		`http_requests_total{action="chat",auth="alice",status="OK",status_code="200"} 2`,
		`http_requests_total{action="chat",auth="bob",status="OK",status_code="200"} 1`,
		`ollama_eval_total{auth="alice",model="llama3",reason="stop"} 62`,
		`ollama_eval_total{auth="bob",model="llama3",reason="stop"} 40`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing per-user line %q\n--- got ---\n%s", want, body)
		}
	}

	// No global auth="" entries should be present — global aggregation
	// is done by the scraper, not the server.
	for _, unwanted := range []string{
		`auth=""`,
	} {
		if strings.Contains(body, unwanted) {
			t.Errorf("metrics output should not contain %q (global aggregation is scraper-side)\n--- got ---\n%s", unwanted, body)
		}
	}
}

func TestMetricsHandlerNewMetrics(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sched := InitScheduler(ctx)
	sched.pendingReqCh = make(chan *LlmRequest, 4)
	sched.loaded = map[string]*runnerRef{}
	_, r := newTestMetricsServer(sched)

	// Record some activity.
	sched.recordHTTPRequests("chat", http.StatusOK, http.StatusText(http.StatusOK), "alice")
	sched.recordHTTPRequests("chat", http.StatusInternalServerError, http.StatusText(http.StatusInternalServerError), "bob")
	sched.recordRequestCompleted("chat", "llama3", "alice")
	sched.recordRequestCompleted("chat", "llama3", "bob")
	sched.recordRequestError("chat", "llama3", "Internal Server Error", "bob")
	sched.recordModelLoad("llama3", "alice")
	sched.recordModelLoad("llama3", "alice")
	sched.recordModelUnload("llama3", "alice")
	sched.recordPromptAndEvalMetricsWithMemory("llama3", "stop", true,
		1500*time.Millisecond, 500*time.Millisecond, 300*time.Millisecond, 450*time.Millisecond,
		131, 62, 120, "alice", 0)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	for _, want := range []string{
		// Config gauges.
		"# HELP ollama_max_loaded_models",
		"# TYPE ollama_max_loaded_models gauge",
		"# HELP ollama_num_parallel",
		"# TYPE ollama_num_parallel gauge",
		// Uptime.
		"# HELP ollama_uptime_seconds",
		"# TYPE ollama_uptime_seconds gauge",
		// Tokens per second.
		"# HELP ollama_tokens_per_second",
		"# TYPE ollama_tokens_per_second gauge",
		// Model load/unload.
		"# HELP ollama_model_load_total",
		"# TYPE ollama_model_load_total counter",
		`ollama_model_load_total{auth="alice",model="llama3"} 2`,
		"# HELP ollama_model_unload_total",
		"# TYPE ollama_model_unload_total counter",
		`ollama_model_unload_total{auth="alice",model="llama3"} 1`,
		// Request errors.
		"# HELP ollama_request_errors_total",
		"# TYPE ollama_request_errors_total counter",
		`ollama_request_errors_total{action="chat",auth="bob",error_type="Internal Server Error",model="llama3"} 1`,
		// Completed requests.
		"# HELP ollama_requests_completed_total",
		"# TYPE ollama_requests_completed_total counter",
		`ollama_requests_completed_total{action="chat",auth="alice",model="llama3"} 1`,
		`ollama_requests_completed_total{action="chat",auth="bob",model="llama3"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing line %q\n--- got ---\n%s", want, body)
		}
	}

	// Verify no GPU metrics when no GPUs cached.
	if strings.Contains(body, "ollama_gpu_memory_total_bytes") {
		t.Errorf("should not emit GPU memory metrics when no GPUs cached")
	}
	if strings.Contains(body, "ollama_context_length") {
		t.Errorf("should not emit context length when no models loaded with options")
	}
}

func TestCloudBuildInfoMetric(t *testing.T) {
	sched := &Scheduler{
		pendingReqCh: make(chan *LlmRequest, 4),
		loaded:       map[string]*runnerRef{},
	}
	_, r := newTestMetricsServer(sched)

	sched.recordCloudBuildInfo("llama3", "abc123", "2026-09-20T10:00:00Z")
	sched.recordCloudBuildInfo("qwen3", "abc123", "2026-09-20T10:00:00Z")
	sched.recordCloudBuildInfo("llama3", "def456", "2026-09-21T10:00:00Z")

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	for _, want := range []string{
		"# HELP ollama_cloud_build_info",
		"# TYPE ollama_cloud_build_info gauge",
		`ollama_cloud_build_info{build_time="2026-09-20T10:00:00Z",commit="abc123",model="llama3"} 1`,
		`ollama_cloud_build_info{build_time="2026-09-20T10:00:00Z",commit="abc123",model="qwen3"} 1`,
		`ollama_cloud_build_info{build_time="2026-09-21T10:00:00Z",commit="def456",model="llama3"} 1`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("metrics output missing %q\n--- got ---\n%s", want, body)
		}
	}
}
