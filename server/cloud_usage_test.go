package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestUsageTracker_NonLegacyPerModelCost(t *testing.T) {
	tr := newUsageTracker()

	resp := usageSummary{
		usageMetrics: usageMetrics{
			RequestCount: 10,
			UsageUSD:     0.05,
		},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 7, UsageUSD: 0.035}},
			{Model: "llama3", usageMetrics: usageMetrics{RequestCount: 3, UsageUSD: 0.015}},
		},
	}

	costs, _, _, _ := tr.update(resp)
	if costs["gpt-4"] != 0.035 {
		t.Fatalf("gpt-4 cost = %v, want 0.035", costs["gpt-4"])
	}
	if costs["llama3"] != 0.015 {
		t.Fatalf("llama3 cost = %v, want 0.015", costs["llama3"])
	}
}

func TestUsageTracker_LegacyRequestCountAttribution(t *testing.T) {
	tr := newUsageTracker()

	// First snapshot: no usage_usd (legacy plan), some requests.
	resp1 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 5, UsageUSD: 0},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 3}},
			{Model: "llama3", usageMetrics: usageMetrics{RequestCount: 2}},
		},
	}
	costs, _, _, _ := tr.update(resp1)
	// No usage data → no cost attributed.
	if len(costs) != 0 {
		t.Fatalf("expected no costs on first snapshot, got %v", costs)
	}

	// Second snapshot: total cost appeared (e.g. plan upgraded), more requests.
	resp2 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 12, UsageUSD: 0.10},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 8, UsageUSD: 0}},
			{Model: "llama3", usageMetrics: usageMetrics{RequestCount: 4, UsageUSD: 0}},
		},
	}
	costs, _, _, _ = tr.update(resp2)
	// usageDelta = 0.10, gpt-4 delta=5, llama3 delta=2, total delta=7
	// gpt-4 share = 5/7, llama3 share = 2/7
	if costs["gpt-4"] == 0 {
		t.Fatalf("gpt-4 cost should be non-zero, got %v", costs["gpt-4"])
	}
	if costs["llama3"] == 0 {
		t.Fatalf("llama3 cost should be non-zero, got %v", costs["llama3"])
	}
	expectedTotal := 0.10
	got := costs["gpt-4"] + costs["llama3"]
	if got < expectedTotal*0.99 || got > expectedTotal*1.01 {
		t.Fatalf("total cost = %v, want ~%v", got, expectedTotal)
	}
}

func TestUsageTracker_LegacySingleModelAttribute(t *testing.T) {
	tr := newUsageTracker()

	resp1 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 5, UsageUSD: 0},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 5}},
		},
	}
	tr.update(resp1)

	resp2 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 8, UsageUSD: 0.03},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 8, UsageUSD: 0}},
		},
	}
	costs, _, _, _ := tr.update(resp2)
	// Only one model changed → entire delta attributed to it.
	if costs["gpt-4"] != 0.03 {
		t.Fatalf("gpt-4 cost = %v, want 0.03", costs["gpt-4"])
	}
}

func TestUsageTracker_ResetDetection(t *testing.T) {
	tr := newUsageTracker()

	resp1 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 100, UsageUSD: 0.50},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 100, UsageUSD: 0.50}},
		},
	}
	tr.update(resp1)
	tr.recordTokens("gpt-4", 1000, 200, 500)

	// Reset: request count drops dramatically.
	resp2 := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 5, UsageUSD: 0.01},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 5, UsageUSD: 0.01}},
		},
	}
	costs, _, _, _ := tr.update(resp2)
	// After reset, per-model cost should be the new usage_usd, not accumulated.
	if costs["gpt-4"] != 0.01 {
		t.Fatalf("after reset, gpt-4 cost = %v, want 0.01", costs["gpt-4"])
	}
}

func TestUsageTracker_CostPerToken(t *testing.T) {
	tr := newUsageTracker()

	tr.recordTokens("gpt-4", 1000, 200, 500)

	resp := usageSummary{
		usageMetrics: usageMetrics{RequestCount: 10, UsageUSD: 0.05},
		Groups: []usageGroup{
			{Model: "gpt-4", usageMetrics: usageMetrics{RequestCount: 10, UsageUSD: 0.05}},
		},
	}
	_, costPerPrompt, costPerEval, costPerToken := tr.update(resp)

	// uncached prompt = 1000 - 200 = 800
	// costPerPrompt = 0.05 / 800
	if costPerPrompt["gpt-4"] == 0 {
		t.Fatal("costPerPrompt should be non-zero")
	}
	// costPerEval = 0.05 / 500
	if costPerEval["gpt-4"] == 0 {
		t.Fatal("costPerEval should be non-zero")
	}
	// costPerToken = 0.05 / (800 + 500)
	if costPerToken["gpt-4"] == 0 {
		t.Fatal("costPerToken should be non-zero")
	}
}

func TestFetchCloudUsageRateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "30")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":"rate limited"}`))
	}))
	defer srv.Close()

	old := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	defer func() { cloudProxyBaseURL = old }()

	oldSign := cloudProxySignRequest
	cloudProxySignRequest = func(_ context.Context, _ *http.Request) error { return nil }
	defer func() { cloudProxySignRequest = oldSign }()

	_, err := fetchCloudUsage(context.Background())
	var rlErr *cloudUsageRateLimitError
	if !errors.As(err, &rlErr) {
		t.Fatalf("expected cloudUsageRateLimitError, got %v", err)
	}
	if rlErr.RetryAfter != 30*time.Second {
		t.Fatalf("RetryAfter = %v, want 30s", rlErr.RetryAfter)
	}
}

func TestFetchCloudUsageGroupByModel(t *testing.T) {
	body := `{
		"range": "7d",
		"scope": "self",
		"granularity": "day",
		"group_by": "model",
		"from": "2026-09-24T00:00:00Z",
		"until": "2026-10-01T02:30:00Z",
		"totals": {
			"request_count": 16,
			"usage_usd": 0.0268,
			"input_tokens": 116000,
			"cached_input_tokens": 60000,
			"output_tokens": 15000,
			"groups": [
				{"model": "glm-5.3-flash", "request_count": 12, "usage_usd": 0.0132, "input_tokens": 96000, "cached_input_tokens": 60000, "output_tokens": 12000},
				{"model": "glm-5.3", "request_count": 2, "usage_usd": 0.01, "input_tokens": 4000, "cached_input_tokens": 0, "output_tokens": 1000},
				{"model": "gpt-oss:120b", "request_count": 2, "usage_usd": 0.0036, "input_tokens": 16000, "cached_input_tokens": 0, "output_tokens": 2000}
			]
		},
		"buckets": []
	}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("group_by") != "model" {
			t.Errorf("group_by query param = %q, want \"model\"", r.URL.Query().Get("group_by"))
		}
		if r.URL.Query().Get("range") != "7d" {
			t.Errorf("range query param = %q, want \"7d\"", r.URL.Query().Get("range"))
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	old := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	defer func() { cloudProxyBaseURL = old }()

	oldSign := cloudProxySignRequest
	cloudProxySignRequest = func(_ context.Context, _ *http.Request) error { return nil }
	defer func() { cloudProxySignRequest = oldSign }()

	usage, err := fetchCloudUsage(context.Background())
	if err != nil {
		t.Fatalf("fetchCloudUsage: %v", err)
	}
	if usage.Range != "7d" {
		t.Fatalf("range = %q, want \"7d\"", usage.Range)
	}
	if usage.GroupBy != "model" {
		t.Fatalf("group_by = %q, want \"model\"", usage.GroupBy)
	}
	if usage.Totals.RequestCount != 16 {
		t.Fatalf("totals.request_count = %d, want 16", usage.Totals.RequestCount)
	}
	if len(usage.Totals.Groups) != 3 {
		t.Fatalf("groups length = %d, want 3", len(usage.Totals.Groups))
	}
	if usage.Totals.Groups[0].Model != "glm-5.3-flash" {
		t.Fatalf("first group model = %q, want \"glm-5.3-flash\"", usage.Totals.Groups[0].Model)
	}
	if usage.Totals.Groups[0].UsageUSD != 0.0132 {
		t.Fatalf("first group usage_usd = %v, want 0.0132", usage.Totals.Groups[0].UsageUSD)
	}
}

func TestFetchCloudUsageLegacyPlan(t *testing.T) {
	body := `{
		"range": "7d",
		"scope": "self",
		"granularity": "day",
		"group_by": "model",
		"from": "2026-09-24T00:00:00Z",
		"until": "2026-10-01T00:00:00Z",
		"totals": {
			"request_count": 15,
			"input_tokens": 106000,
			"cached_input_tokens": 46000,
			"output_tokens": 13600,
			"groups": [
				{"model": "glm-5.3-flash", "request_count": 12, "input_tokens": 96000, "cached_input_tokens": 46000, "output_tokens": 12000},
				{"model": "glm-5.3", "request_count": 3, "input_tokens": 10000, "cached_input_tokens": 0, "output_tokens": 1600}
			]
		},
		"buckets": []
	}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	old := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	defer func() { cloudProxyBaseURL = old }()

	oldSign := cloudProxySignRequest
	cloudProxySignRequest = func(_ context.Context, _ *http.Request) error { return nil }
	defer func() { cloudProxySignRequest = oldSign }()

	usage, err := fetchCloudUsage(context.Background())
	if err != nil {
		t.Fatalf("fetchCloudUsage: %v", err)
	}
	// Legacy plan: no usage_usd in totals or groups.
	if usage.Totals.UsageUSD != 0 {
		t.Fatalf("legacy plan totals.usage_usd = %v, want 0", usage.Totals.UsageUSD)
	}
	for i, g := range usage.Totals.Groups {
		if g.UsageUSD != 0 {
			t.Fatalf("legacy plan group[%d].usage_usd = %v, want 0", i, g.UsageUSD)
		}
	}
	// Request counts and tokens should still be present.
	if usage.Totals.Groups[0].RequestCount != 12 {
		t.Fatalf("group[0].request_count = %d, want 12", usage.Totals.Groups[0].RequestCount)
	}
	if usage.Totals.Groups[0].InputTokens != 96000 {
		t.Fatalf("group[0].input_tokens = %d, want 96000", usage.Totals.Groups[0].InputTokens)
	}
}

func TestRecordCloudUsageMetrics(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.usageTracker = newUsageTracker()

	usage := &cloudUsageResponse{
		Range:   "7d",
		GroupBy: "model",
		Totals: usageSummary{
			usageMetrics: usageMetrics{
				RequestCount: 16,
				UsageUSD:     0.0268,
			},
			Groups: []usageGroup{
				{Model: "glm-5.3-flash", usageMetrics: usageMetrics{RequestCount: 12, UsageUSD: 0.0132, InputTokens: 96000, CachedInputTokens: 60000, OutputTokens: 12000}},
				{Model: "gpt-oss:120b", usageMetrics: usageMetrics{RequestCount: 2, UsageUSD: 0.0036, InputTokens: 16000, CachedInputTokens: 0, OutputTokens: 2000}},
			},
		},
	}

	s.recordCloudUsage(usage)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	for _, want := range []string{
		`ollama_cloud_usage_total_cost{range="7d"} 0.0268`,
		`ollama_cloud_usage_total_requests{range="7d"} 16`,
		`ollama_cloud_usage_requests{model="glm-5.3-flash",range="7d"} 12`,
		`ollama_cloud_usage_input_tokens{model="glm-5.3-flash"} 96000`,
		`ollama_cloud_usage_output_tokens{model="glm-5.3-flash"} 12000`,
		`ollama_cloud_usage_cost{model="glm-5.3-flash"} 0.0132`,
	} {
		if !contains(body, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
}

func TestRecordCloudUsageLegacyPlan(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.usageTracker = newUsageTracker()

	usage := &cloudUsageResponse{
		Range:   "7d",
		GroupBy: "model",
		Totals: usageSummary{
			usageMetrics: usageMetrics{RequestCount: 15},
			Groups: []usageGroup{
				{Model: "glm-5.3-flash", usageMetrics: usageMetrics{RequestCount: 12, InputTokens: 96000, CachedInputTokens: 46000, OutputTokens: 12000}},
			},
		},
	}

	s.recordCloudUsage(usage)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	// Legacy: total cost should be 0, but request counts and tokens present.
	if !contains(body, `ollama_cloud_usage_total_cost{range="7d"} 0`) {
		t.Errorf("legacy plan should show 0 total cost")
	}
	if !contains(body, `ollama_cloud_usage_requests{model="glm-5.3-flash",range="7d"} 12`) {
		t.Errorf("legacy plan should still show request counts")
	}
	if !contains(body, `ollama_cloud_usage_input_tokens{model="glm-5.3-flash"} 96000`) {
		t.Errorf("legacy plan should still show input tokens")
	}
}

func contains(s, substr string) bool {
	return indexOf(s, substr) >= 0
}

func indexOf(s, substr string) int {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}

func TestFetchCloudUsageRangeDefault(t *testing.T) {
	var capturedRange string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedRange = r.URL.Query().Get("range")
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"range":"7d","scope":"self","granularity":"day","from":"2026-09-24T00:00:00Z","until":"2026-10-01T00:00:00Z","totals":{"request_count":0},"buckets":[]}`))
	}))
	defer srv.Close()

	old := cloudProxyBaseURL
	cloudProxyBaseURL = srv.URL
	defer func() { cloudProxyBaseURL = old }()

	oldSign := cloudProxySignRequest
	cloudProxySignRequest = func(_ context.Context, _ *http.Request) error { return nil }
	defer func() { cloudProxySignRequest = oldSign }()

	_, _ = fetchCloudUsage(context.Background())
	if capturedRange != "7d" {
		t.Fatalf("default range = %q, want \"7d\"", capturedRange)
	}
}

func TestCloudUsageResponseUnmarshal(t *testing.T) {
	raw := `{
		"range": "24h",
		"scope": "self",
		"granularity": "hour",
		"group_by": "model",
		"from": "2026-09-30T02:00:00Z",
		"until": "2026-10-01T02:30:00Z",
		"totals": {
			"request_count": 15,
			"usage_usd": 0.01718,
			"input_tokens": 106000,
			"cached_input_tokens": 46000,
			"output_tokens": 13600,
			"groups": [
				{"model": "glm-5.3-flash", "request_count": 12, "usage_usd": 0.0132},
				{"model": "glm-5.3", "request_count": 2, "usage_usd": 0.01},
				{"model": "gpt-oss:120b", "request_count": 1, "usage_usd": 0.001}
			]
		},
		"buckets": [
			{
				"from": "2026-09-30T12:00:00Z",
				"until": "2026-09-30T13:00:00Z",
				"request_count": 4,
				"usage_usd": 0.0052,
				"groups": [
					{"model": "glm-5.3-flash", "request_count": 4, "usage_usd": 0.0052}
				]
			},
			{
				"from": "2026-10-01T02:00:00Z",
				"until": "2026-10-01T02:30:00Z",
				"partial": true,
				"request_count": 3,
				"usage_usd": 0.00318
			}
		]
	}`

	var resp cloudUsageResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Range != "24h" {
		t.Fatalf("range = %q, want \"24h\"", resp.Range)
	}
	if resp.Totals.RequestCount != 15 {
		t.Fatalf("totals.request_count = %d, want 15", resp.Totals.RequestCount)
	}
	if len(resp.Totals.Groups) != 3 {
		t.Fatalf("groups length = %d, want 3", len(resp.Totals.Groups))
	}
	if len(resp.Buckets) != 2 {
		t.Fatalf("buckets length = %d, want 2", len(resp.Buckets))
	}
	if !resp.Buckets[1].Partial {
		t.Fatal("bucket[1] should be partial")
	}
}
