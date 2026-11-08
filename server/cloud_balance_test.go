package server

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func approxEqual(a, b float64) bool {
	return math.Abs(a-b) < 1e-9
}

func TestCloudBalanceResponseUnmarshal(t *testing.T) {
	raw := `{
		"included": {
			"balance_usd": 72.5,
			"allowance_usd": 100,
			"period": {
				"from": "2026-09-15T09:30:00Z",
				"until": "2026-10-15T09:30:00Z"
			}
		},
		"purchased": {
			"balance_usd": 25
		}
	}`

	var resp cloudBalanceResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if resp.Included.BalanceUSD != 72.5 {
		t.Fatalf("included.balance_usd = %v, want 72.5", resp.Included.BalanceUSD)
	}
	if resp.Included.AllowanceUSD != 100 {
		t.Fatalf("included.allowance_usd = %v, want 100", resp.Included.AllowanceUSD)
	}
	if resp.Purchased.BalanceUSD != 25 {
		t.Fatalf("purchased.balance_usd = %v, want 25", resp.Purchased.BalanceUSD)
	}
	if resp.isLegacy() {
		t.Fatal("non-legacy response should not be detected as legacy")
	}
	if resp.totalBalance() != 97.5 {
		t.Fatalf("totalBalance = %v, want 97.5", resp.totalBalance())
	}
}

func TestCloudBalanceLegacyResponseUnmarshal(t *testing.T) {
	raw := `{
		"included": {
			"session": {
				"remaining_percent": 75,
				"resets_at": "2026-10-01T07:00:00Z"
			},
			"weekly": {
				"remaining_percent": 40,
				"resets_at": "2026-10-05T00:00:00Z"
			}
		},
		"purchased": {
			"balance_usd": 25
		}
	}`

	var resp cloudBalanceResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if !resp.isLegacy() {
		t.Fatal("legacy response should be detected as legacy")
	}
	if resp.Included.Session == nil {
		t.Fatal("session should be non-nil")
	}
	if resp.Included.Session.RemainingPercent != 75 {
		t.Fatalf("session.remaining_percent = %v, want 75", resp.Included.Session.RemainingPercent)
	}
	if resp.Included.Weekly == nil {
		t.Fatal("weekly should be non-nil")
	}
	if resp.Included.Weekly.RemainingPercent != 40 {
		t.Fatalf("weekly.remaining_percent = %v, want 40", resp.Included.Weekly.RemainingPercent)
	}
}

func TestFetchCloudBalance(t *testing.T) {
	body := `{
		"included": {
			"balance_usd": 72.5,
			"allowance_usd": 100,
			"period": {
				"from": "2026-09-15T09:30:00Z",
				"until": "2026-10-15T09:30:00Z"
			}
		},
		"purchased": {
			"balance_usd": 25
		}
	}`

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/balance" {
			t.Errorf("path = %q, want /api/balance", r.URL.Path)
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

	balance, err := fetchCloudBalance(context.Background())
	if err != nil {
		t.Fatalf("fetchCloudBalance: %v", err)
	}
	if balance.Included.BalanceUSD != 72.5 {
		t.Fatalf("included.balance_usd = %v, want 72.5", balance.Included.BalanceUSD)
	}
	if balance.Purchased.BalanceUSD != 25 {
		t.Fatalf("purchased.balance_usd = %v, want 25", balance.Purchased.BalanceUSD)
	}
}

func TestFetchCloudBalanceRateLimit(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "45")
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

	_, err := fetchCloudBalance(context.Background())
	var rlErr *cloudRateLimitError
	if !errors.As(err, &rlErr) {
		t.Fatalf("expected cloudRateLimitError, got %v", err)
	}
	if rlErr.RetryAfter != 45*time.Second {
		t.Fatalf("RetryAfter = %v, want 45s", rlErr.RetryAfter)
	}
}

func TestBalanceTracker_SingleModelAttribute(t *testing.T) {
	tr := newBalanceTracker()

	// First fetch: no previous balance, just stores it.
	costs := tr.update(100.0)
	if len(costs) != 0 {
		t.Fatalf("expected no costs on first update, got %v", costs)
	}

	// One request for model-a.
	tr.recordPendingRequest("model-a")

	// Balance dropped by 0.05.
	costs = tr.update(99.95)
	if !approxEqual(costs["model-a"], 0.05) {
		t.Fatalf("model-a cost = %v, want ~0.05", costs["model-a"])
	}
}

func TestBalanceTracker_MultiModelProportionalAttribution(t *testing.T) {
	tr := newBalanceTracker()
	tr.update(100.0)

	// 3 requests for model-a, 1 for model-b.
	tr.recordPendingRequest("model-a")
	tr.recordPendingRequest("model-a")
	tr.recordPendingRequest("model-a")
	tr.recordPendingRequest("model-b")

	// Balance dropped by 0.08.
	costs := tr.update(99.92)
	// model-a share = 3/4, model-b share = 1/4
	if costs["model-a"] < 0.059 || costs["model-a"] > 0.061 {
		t.Fatalf("model-a cost = %v, want ~0.06", costs["model-a"])
	}
	if costs["model-b"] < 0.019 || costs["model-b"] > 0.021 {
		t.Fatalf("model-b cost = %v, want ~0.02", costs["model-b"])
	}
}

func TestBalanceTracker_ResetDetection(t *testing.T) {
	tr := newBalanceTracker()
	tr.update(100.0)
	tr.recordPendingRequest("model-a")
	tr.update(99.90) // cost 0.10

	// Balance jumps up — period reset.
	costs := tr.update(100.0)
	if costs["model-a"] != 0 {
		t.Fatalf("after reset, model-a cost = %v, want 0", costs["model-a"])
	}

	// New requests after reset.
	tr.recordPendingRequest("model-a")
	costs = tr.update(99.95)
	if !approxEqual(costs["model-a"], 0.05) {
		t.Fatalf("after reset + new request, model-a cost = %v, want ~0.05", costs["model-a"])
	}
}

func TestBalanceTracker_AccumulationAcrossFetches(t *testing.T) {
	tr := newBalanceTracker()
	tr.update(100.0)

	tr.recordPendingRequest("model-a")
	costs := tr.update(99.97)
	if !approxEqual(costs["model-a"], 0.03) {
		t.Fatalf("first fetch: model-a cost = %v, want ~0.03", costs["model-a"])
	}

	tr.recordPendingRequest("model-a")
	costs = tr.update(99.94)
	if !approxEqual(costs["model-a"], 0.06) {
		t.Fatalf("second fetch: model-a cost = %v, want ~0.06 (accumulated)", costs["model-a"])
	}
}

func TestRecordCloudBalanceMetrics(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.balanceTracker = newBalanceTracker()

	balance := &cloudBalanceResponse{
		Included: includedBalance{
			BalanceUSD:   72.5,
			AllowanceUSD: 100,
			Period: &balancePeriod{
				From:  time.Date(2026, 9, 15, 9, 30, 0, 0, time.UTC),
				Until: time.Date(2026, 10, 15, 9, 30, 0, 0, time.UTC),
			},
		},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}

	s.recordCloudBalance(balance)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	for _, want := range []string{
		`ollama_cloud_included_balance_usd 72.5`,
		`ollama_cloud_included_allowance_usd 100`,
		`ollama_cloud_purchased_balance_usd 25`,
	} {
		if !contains(body, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
}

func TestRecordCloudBalanceLegacyMetrics(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.balanceTracker = newBalanceTracker()

	balance := &cloudBalanceResponse{
		Included: includedBalance{
			Session: &legacyBalanceLimit{RemainingPercent: 75, ResetsAt: time.Date(2026, 10, 1, 7, 0, 0, 0, time.UTC)},
			Weekly:  &legacyBalanceLimit{RemainingPercent: 40, ResetsAt: time.Date(2026, 10, 5, 0, 0, 0, 0, time.UTC)},
		},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}

	s.recordCloudBalance(balance)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	for _, want := range []string{
		`ollama_cloud_session_remaining_percent 75`,
		`ollama_cloud_weekly_remaining_percent 40`,
		`ollama_cloud_purchased_balance_usd 25`,
	} {
		if !contains(body, want) {
			t.Errorf("metrics missing %q", want)
		}
	}
}

func TestRecordCloudBalanceCostFromDelta(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.balanceTracker = newBalanceTracker()

	// First balance fetch — establishes baseline.
	balance1 := &cloudBalanceResponse{
		Included:  includedBalance{BalanceUSD: 72.5, AllowanceUSD: 100},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}
	s.recordCloudBalance(balance1)

	// Simulate a request for model-a.
	s.balanceTracker.recordPendingRequest("model-a")

	// Second balance fetch — balance dropped by 0.03.
	balance2 := &cloudBalanceResponse{
		Included:  includedBalance{BalanceUSD: 72.47, AllowanceUSD: 100},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}
	s.recordCloudBalance(balance2)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	if !contains(body, `ollama_cloud_balance_cost{model="model-a"} 0.03`) {
		t.Errorf("metrics missing balance cost for model-a\nbody:\n%s", body)
	}
}

func TestBalanceTracker_LegacySessionUsage(t *testing.T) {
	tr := newBalanceTracker()

	// First fetch: establishes baseline.
	sessionUsage, _ := tr.updateLegacy(80, 50)
	if len(sessionUsage) != 0 {
		t.Fatalf("expected no usage on first update, got %v", sessionUsage)
	}

	// One request for model-a.
	tr.recordPendingRequest("model-a")

	// Session dropped by 5 percentage points.
	sessionUsage, _ = tr.updateLegacy(75, 50)
	if !approxEqual(sessionUsage["model-a"], 5) {
		t.Fatalf("model-a session usage = %v, want ~5", sessionUsage["model-a"])
	}
}

func TestBalanceTracker_LegacyWeeklyUsage(t *testing.T) {
	tr := newBalanceTracker()
	tr.updateLegacy(80, 50)

	tr.recordPendingRequest("model-b")

	// Weekly dropped by 10 percentage points.
	_, weeklyUsage := tr.updateLegacy(80, 40)
	if !approxEqual(weeklyUsage["model-b"], 10) {
		t.Fatalf("model-b weekly usage = %v, want ~10", weeklyUsage["model-b"])
	}
}

func TestBalanceTracker_LegacyBothSessionAndWeekly(t *testing.T) {
	tr := newBalanceTracker()
	tr.updateLegacy(90, 60)

	// 2 requests for model-a, 1 for model-b.
	tr.recordPendingRequest("model-a")
	tr.recordPendingRequest("model-a")
	tr.recordPendingRequest("model-b")

	// Session dropped 3, weekly dropped 6.
	sessionUsage, weeklyUsage := tr.updateLegacy(87, 54)
	// model-a share = 2/3, model-b share = 1/3
	if !approxEqual(sessionUsage["model-a"], 2) {
		t.Fatalf("model-a session usage = %v, want ~2", sessionUsage["model-a"])
	}
	if !approxEqual(sessionUsage["model-b"], 1) {
		t.Fatalf("model-b session usage = %v, want ~1", sessionUsage["model-b"])
	}
	if !approxEqual(weeklyUsage["model-a"], 4) {
		t.Fatalf("model-a weekly usage = %v, want ~4", weeklyUsage["model-a"])
	}
	if !approxEqual(weeklyUsage["model-b"], 2) {
		t.Fatalf("model-b weekly usage = %v, want ~2", weeklyUsage["model-b"])
	}
}

func TestBalanceTracker_LegacySessionReset(t *testing.T) {
	tr := newBalanceTracker()
	tr.updateLegacy(80, 50)
	tr.recordPendingRequest("model-a")
	tr.updateLegacy(75, 45) // consumed 5 session, 5 weekly

	// Session resets (percentage jumps up).
	sessionUsage, _ := tr.updateLegacy(100, 45)
	if sessionUsage["model-a"] != 0 {
		t.Fatalf("after session reset, model-a session usage = %v, want 0", sessionUsage["model-a"])
	}

	// Weekly continues accumulating.
	_, weeklyUsage := tr.updateLegacy(100, 45)
	// No new pending requests since last fetch, so no new consumption.
	if !approxEqual(weeklyUsage["model-a"], 5) {
		t.Fatalf("weekly should still show accumulated 5, got %v", weeklyUsage["model-a"])
	}
}

func TestBalanceTracker_LegacyWeeklyReset(t *testing.T) {
	tr := newBalanceTracker()
	tr.updateLegacy(80, 50)
	tr.recordPendingRequest("model-a")
	tr.updateLegacy(75, 40) // consumed 5 session, 10 weekly

	// Weekly resets (percentage jumps up).
	_, weeklyUsage := tr.updateLegacy(75, 100)
	if weeklyUsage["model-a"] != 0 {
		t.Fatalf("after weekly reset, model-a weekly usage = %v, want 0", weeklyUsage["model-a"])
	}
}

func TestBalanceTracker_LegacyAccumulationAcrossFetches(t *testing.T) {
	tr := newBalanceTracker()
	tr.updateLegacy(90, 70)

	tr.recordPendingRequest("model-a")
	sessionUsage, weeklyUsage := tr.updateLegacy(85, 65)
	if !approxEqual(sessionUsage["model-a"], 5) {
		t.Fatalf("first fetch: session usage = %v, want ~5", sessionUsage["model-a"])
	}
	if !approxEqual(weeklyUsage["model-a"], 5) {
		t.Fatalf("first fetch: weekly usage = %v, want ~5", weeklyUsage["model-a"])
	}

	tr.recordPendingRequest("model-a")
	sessionUsage, weeklyUsage = tr.updateLegacy(80, 60)
	if !approxEqual(sessionUsage["model-a"], 10) {
		t.Fatalf("second fetch: session usage = %v, want ~10 (accumulated)", sessionUsage["model-a"])
	}
	if !approxEqual(weeklyUsage["model-a"], 10) {
		t.Fatalf("second fetch: weekly usage = %v, want ~10 (accumulated)", weeklyUsage["model-a"])
	}
}

func TestRecordCloudBalanceLegacyUsageMetrics(t *testing.T) {
	s, r := newTestMetricsServer(nil)
	s.balanceTracker = newBalanceTracker()

	// First balance fetch — establishes baseline.
	balance1 := &cloudBalanceResponse{
		Included: includedBalance{
			Session: &legacyBalanceLimit{RemainingPercent: 80, ResetsAt: time.Date(2026, 10, 1, 7, 0, 0, 0, time.UTC)},
			Weekly:  &legacyBalanceLimit{RemainingPercent: 50, ResetsAt: time.Date(2026, 10, 5, 0, 0, 0, 0, time.UTC)},
		},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}
	s.recordCloudBalance(balance1)

	// Simulate a request for model-a.
	s.balanceTracker.recordPendingRequest("model-a")

	// Second balance fetch — session dropped 5, weekly dropped 10.
	balance2 := &cloudBalanceResponse{
		Included: includedBalance{
			Session: &legacyBalanceLimit{RemainingPercent: 75, ResetsAt: time.Date(2026, 10, 1, 7, 0, 0, 0, time.UTC)},
			Weekly:  &legacyBalanceLimit{RemainingPercent: 40, ResetsAt: time.Date(2026, 10, 5, 0, 0, 0, 0, time.UTC)},
		},
		Purchased: purchasedBalance{BalanceUSD: 25},
	}
	s.recordCloudBalance(balance2)

	w := httptest.NewRecorder()
	r.ServeHTTP(w, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	body := w.Body.String()

	if !contains(body, `ollama_cloud_session_usage_consumed{model="model-a"} 5`) {
		t.Errorf("metrics missing session usage for model-a\nbody:\n%s", body)
	}
	if !contains(body, `ollama_cloud_weekly_usage_consumed{model="model-a"} 10`) {
		t.Errorf("metrics missing weekly usage for model-a\nbody:\n%s", body)
	}
	if !contains(body, `ollama_cloud_session_remaining_percent 75`) {
		t.Errorf("metrics missing session remaining percent\nbody:\n%s", body)
	}
	if !contains(body, `ollama_cloud_weekly_remaining_percent 40`) {
		t.Errorf("metrics missing weekly remaining percent\nbody:\n%s", body)
	}
}
