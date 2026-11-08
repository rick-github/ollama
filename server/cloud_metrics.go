package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/version"
)

type usageMetrics struct {
	RequestCount      int     `json:"request_count"`
	UsageUSD          float64 `json:"usage_usd,omitempty"`
	InputTokens       int     `json:"input_tokens,omitempty"`
	CachedInputTokens int     `json:"cached_input_tokens,omitempty"`
	OutputTokens      int     `json:"output_tokens,omitempty"`
}

// usageGroup is a per-model breakdown within a usage summary. It embeds
// usageMetrics so the JSON flattens request_count, usage_usd, etc.
type usageGroup struct {
	usageMetrics
	Model string `json:"model"`
}

// usageSummary is the aggregate usage for a range or bucket, optionally
// broken down by model via groups.
type usageSummary struct {
	usageMetrics
	Groups    []usageGroup  `json:"groups,omitempty"`
	Remainder *usageMetrics `json:"remainder,omitempty"`
}

// usageBucket is one time-slice within the range. The /api/usage endpoint
// returns per-bucket breakdowns (hourly for 24h, daily for 7d/30d), but
// they are currently unused — monitoring relies on Prometheus scrape
// intervals for time resolution rather than the API's own buckets. The
// type is retained for future use.
type usageBucket struct {
	usageSummary
	From    time.Time `json:"from"`
	Until   time.Time `json:"until"`
	Partial bool      `json:"partial,omitempty"`
}

// cloudUsageResponse is the JSON structure returned by /api/usage.
// The Buckets field is decoded but not used for monitoring (see
// usageBucket comment above).
type cloudUsageResponse struct {
	Range       string        `json:"range"`
	Scope       string        `json:"scope"`
	Granularity string        `json:"granularity"`
	GroupBy     string        `json:"group_by,omitempty"`
	From        time.Time     `json:"from"`
	Until       time.Time     `json:"until"`
	Totals      usageSummary  `json:"totals"`
	Buckets     []usageBucket `json:"buckets,omitempty"`
}

// usageTracker maintains the previous /api/usage snapshot to compute
// per-model cost deltas between fetches. When the API returns per-model
// usage_usd (non-legacy plans), costs are read directly from the groups.
// When usage_usd is absent (legacy plans), the tracker falls back to
// proportional attribution of the total cost delta using per-model
// request count deltas.
//
// In addition to cost, the tracker accumulates per-model prompt, cached
// prompt, and eval token counts from local cloud-proxied responses to
// compute cost-per-token rates and to expose token count gauges that the
// pricing scraper can join with price gauges to estimate cost for legacy
// plans.
type usageTracker struct {
	mu                      sync.Mutex
	prevUsage               float64
	prevCounts              map[string]int
	modelCosts              map[string]float64
	modelPromptTokens       map[string]int
	modelCachedPromptTokens map[string]int
	modelEvalTokens         map[string]int
	// hasPerModelCost is true once we've seen non-zero usage_usd in a
	// group, indicating a non-legacy plan. When true, cost is read
	// directly from groups and delta-based attribution is skipped.
	hasPerModelCost bool
	// backoffUntil is the time before which usage fetches should be
	// skipped due to a prior 429 response.
	backoffUntil time.Time
}

func newUsageTracker() *usageTracker {
	return &usageTracker{
		prevCounts:              make(map[string]int),
		modelCosts:              make(map[string]float64),
		modelPromptTokens:       make(map[string]int),
		modelCachedPromptTokens: make(map[string]int),
		modelEvalTokens:         make(map[string]int),
	}
}

// recordTokens accumulates prompt, cached prompt, and eval token counts per
// model from the cloud response. Called alongside recordPromptAndEvalMetrics
// for each cloud-proxied request.
func (t *usageTracker) recordTokens(model string, promptEvalCount, promptEvalCachedCount, evalCount int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.modelPromptTokens[model] += promptEvalCount
	t.modelCachedPromptTokens[model] += promptEvalCachedCount
	t.modelEvalTokens[model] += evalCount
}

// update computes per-model costs from a new usage snapshot. When the
// API provides per-model usage_usd (non-legacy plans), costs are read
// directly from groups. When usage_usd is absent (legacy plans), the
// total usage delta is attributed proportionally to models based on
// request count deltas.
func (t *usageTracker) update(summary usageSummary) (costs map[string]float64, costPerPromptToken, costPerEvalToken, costPerToken map[string]float64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	totalUsage := summary.UsageUSD
	totalRequests := summary.RequestCount

	// Detect reset: if usage or request count dropped significantly,
	// reset accumulated state.
	if (totalUsage < t.prevUsage*0.5 && t.prevUsage > 0) ||
		(totalRequests < t.prevCounts[""]/2 && t.prevCounts[""] > 0) {
		t.modelCosts = make(map[string]float64)
		t.prevCounts = make(map[string]int)
		t.modelPromptTokens = make(map[string]int)
		t.modelCachedPromptTokens = make(map[string]int)
		t.modelEvalTokens = make(map[string]int)
		t.hasPerModelCost = false
	}

	// Check whether any group has non-zero usage_usd (non-legacy plan).
	for _, g := range summary.Groups {
		if g.UsageUSD > 0 {
			t.hasPerModelCost = true
			break
		}
	}

	if t.hasPerModelCost && len(summary.Groups) > 0 {
		// Non-legacy: use per-model usage_usd directly.
		t.modelCosts = make(map[string]float64, len(summary.Groups))
		for _, g := range summary.Groups {
			t.modelCosts[g.Model] = g.UsageUSD
		}
	} else {
		// Legacy or no per-model cost: attribute total usage delta
		// proportionally by request count deltas.
		usageDelta := totalUsage - t.prevUsage
		if usageDelta < 0 {
			usageDelta = 0
		}

		curCounts := make(map[string]int, len(summary.Groups))
		for _, g := range summary.Groups {
			curCounts[g.Model] = g.RequestCount
		}

		type modelDelta struct {
			name  string
			delta int
		}
		var deltas []modelDelta
		totalReqDelta := 0
		for name, cur := range curCounts {
			prev := t.prevCounts[name]
			d := cur - prev
			if d > 0 {
				deltas = append(deltas, modelDelta{name, d})
				totalReqDelta += d
			}
		}

		if usageDelta > 0 && totalReqDelta > 0 {
			if len(deltas) == 1 {
				t.modelCosts[deltas[0].name] += usageDelta
			} else {
				for _, d := range deltas {
					share := float64(d.delta) / float64(totalReqDelta)
					t.modelCosts[d.name] += usageDelta * share
				}
			}
		}

		t.prevCounts = curCounts
	}

	t.prevUsage = totalUsage
	t.prevCounts[""] = totalRequests

	// Return copies of current costs.
	costs = make(map[string]float64, len(t.modelCosts))
	for k, v := range t.modelCosts {
		costs[k] = v
	}

	// Compute cost-per-token for each model that has token data.
	costPerPromptToken = make(map[string]float64, len(t.modelCosts))
	costPerEvalToken = make(map[string]float64, len(t.modelCosts))
	costPerToken = make(map[string]float64, len(t.modelCosts))
	for model, cost := range t.modelCosts {
		promptTokens := t.modelPromptTokens[model]
		cachedTokens := t.modelCachedPromptTokens[model]
		uncachedPromptTokens := promptTokens - cachedTokens
		evalTokens := t.modelEvalTokens[model]
		if uncachedPromptTokens > 0 {
			costPerPromptToken[model] = cost / float64(uncachedPromptTokens)
		}
		if evalTokens > 0 {
			costPerEvalToken[model] = cost / float64(evalTokens)
		}
		totalChargedTokens := uncachedPromptTokens + evalTokens
		if totalChargedTokens > 0 {
			costPerToken[model] = cost / float64(totalChargedTokens)
		}
	}
	return costs, costPerPromptToken, costPerEvalToken, costPerToken
}

// tokenCounts returns a snapshot of per-model cumulative token counts.
func (t *usageTracker) tokenCounts() (prompt, cached, eval map[string]int) {
	t.mu.Lock()
	defer t.mu.Unlock()
	prompt = make(map[string]int, len(t.modelPromptTokens))
	cached = make(map[string]int, len(t.modelCachedPromptTokens))
	eval = make(map[string]int, len(t.modelEvalTokens))
	for k, v := range t.modelPromptTokens {
		prompt[k] = v
	}
	for k, v := range t.modelCachedPromptTokens {
		cached[k] = v
	}
	for k, v := range t.modelEvalTokens {
		eval[k] = v
	}
	return
}

// isRateLimited returns true if the usage API is still in a 429 backoff period.
func (t *usageTracker) isRateLimited() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return time.Now().Before(t.backoffUntil)
}

// setBackoff records the 429 retry-after duration so subsequent fetches
// are skipped until the backoff expires.
func (t *usageTracker) setBackoff(d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.backoffUntil = time.Now().Add(d)
}

// cloudRateLimitError is returned by cloud API fetchers when the cloud
// service responds with 429. The RetryAfter field holds the number of
// seconds the server asked us to wait.
type cloudRateLimitError struct {
	RetryAfter time.Duration
}

func (e *cloudRateLimitError) Error() string {
	return fmt.Sprintf("cloud API rate limited, retry after %s", e.RetryAfter)
}

// cloudUsageRateLimitError is kept for backward compatibility with tests
// that check for this specific type. It is an alias for cloudRateLimitError.
type cloudUsageRateLimitError = cloudRateLimitError

// fetchCloudUsage fetches /api/usage from the cloud service with
// group_by=model to get per-model breakdowns. The default range is 7d.
func fetchCloudUsage(ctx context.Context) (*cloudUsageResponse, error) {
	return fetchCloudUsageRange(ctx, "7d")
}

// fetchCloudUsageRange fetches /api/usage for the given range (24h, 7d, 30d)
// with group_by=model.
func fetchCloudUsageRange(ctx context.Context, r string) (*cloudUsageResponse, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		return nil, err
	}
	q := url.Values{}
	q.Set("range", r)
	q.Set("group_by", "model")
	targetURL := baseURL.ResolveReference(&url.URL{
		Path:     "/api/usage",
		RawQuery: q.Encode(),
	})

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, targetURL.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		req.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}

	if err := cloudProxySignRequest(req.Context(), req); err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := 60 * time.Second
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
				retryAfter = time.Duration(secs) * time.Second
			}
		}
		return nil, &cloudUsageRateLimitError{RetryAfter: retryAfter}
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, fmt.Errorf("cloud usage fetch failed: %s", resp.Status)
	}

	var usage cloudUsageResponse
	if err := json.NewDecoder(resp.Body).Decode(&usage); err != nil {
		return nil, err
	}
	return &usage, nil
}

// recordCloudUsage updates prometheus gauges from a /api/usage response.
func (s *Server) recordCloudUsage(usage *cloudUsageResponse) {
	if usage == nil || s.sched == nil || s.sched.metrics == nil {
		return
	}

	// Total cost and request count for the range.
	s.sched.metrics.cloudUsageTotalCost.WithLabelValues(usage.Range).Set(usage.Totals.UsageUSD)
	s.sched.metrics.cloudUsageTotalRequests.WithLabelValues(usage.Range).Set(float64(usage.Totals.RequestCount))

	// Per-model request counts and token counts from API groups.
	for _, g := range usage.Totals.Groups {
		model := strings.ToLower(g.Model)
		s.sched.metrics.cloudUsageRequests.WithLabelValues(model, usage.Range).Set(float64(g.RequestCount))
		s.sched.metrics.cloudUsageInputTokens.WithLabelValues(model).Set(float64(g.InputTokens))
		s.sched.metrics.cloudUsageCachedInputTokens.WithLabelValues(model).Set(float64(g.CachedInputTokens))
		s.sched.metrics.cloudUsageOutputTokens.WithLabelValues(model).Set(float64(g.OutputTokens))
	}

	// Compute per-model cost from the usage snapshot.
	costs, costPerPromptToken, costPerEvalToken, costPerToken := s.usageTracker.update(usage.Totals)
	for model, cost := range costs {
		s.sched.metrics.cloudUsageCost.WithLabelValues(model).Set(cost)
	}
	for model, cpt := range costPerPromptToken {
		s.sched.metrics.cloudCostPerPromptToken.WithLabelValues(model).Set(cpt)
	}
	for model, cpt := range costPerEvalToken {
		s.sched.metrics.cloudCostPerEvalToken.WithLabelValues(model).Set(cpt)
	}
	for model, cpt := range costPerToken {
		s.sched.metrics.cloudCostPerToken.WithLabelValues(model).Set(cpt)
	}

	// Set cumulative token count gauges from local tracking.
	prompt, cached, eval := s.usageTracker.tokenCounts()
	for model, count := range prompt {
		s.sched.metrics.cloudPromptTokens.WithLabelValues(model).Set(float64(count))
	}
	for model, count := range cached {
		s.sched.metrics.cloudCachedPromptTokens.WithLabelValues(model).Set(float64(count))
	}
	for model, count := range eval {
		s.sched.metrics.cloudEvalTokens.WithLabelValues(model).Set(float64(count))
	}
}

// fetchAndRecordCloudUsage is the shared entry point for both on-demand and
// interval modes. It fetches /api/usage and records the gauges. Returns the
// rate-limit retry duration if the API returned 429.
func (s *Server) fetchAndRecordCloudUsage(ctx context.Context) time.Duration {
	if s.usageTracker != nil && s.usageTracker.isRateLimited() {
		return 0
	}
	usage, err := fetchCloudUsage(ctx)
	if err != nil {
		var rlErr *cloudUsageRateLimitError
		if errors.As(err, &rlErr) {
			slog.Debug("cloud usage rate limited", "retry_after", rlErr.RetryAfter)
			if s.usageTracker != nil {
				s.usageTracker.setBackoff(rlErr.RetryAfter)
			}
			return rlErr.RetryAfter
		}
		slog.Debug("cloud usage fetch failed", "error", err)
		return 0
	}
	s.recordCloudUsage(usage)
	return 0
}

// startCloudUsageFetcher starts a background goroutine that polls /api/usage
// at the configured interval. It is used when OLLAMA_METRICS_CLOUD_USAGE=interval.
// When the API returns 429, the next fetch is delayed by the Retry-After
// duration returned by the server.
func (s *Server) startCloudUsageFetcher(ctx context.Context) {
	interval := time.Duration(envconfig.MetricsCloudUsageInterval()) * time.Second
	if interval < 5*time.Second {
		interval = 5 * time.Second
	}
	go func() {
		delay := s.fetchAndRecordCloudMetrics(ctx)
		for {
			if delay > 0 {
				select {
				case <-ctx.Done():
					return
				case <-time.After(delay):
				}
				delay = 0
				continue
			}
			select {
			case <-ctx.Done():
				return
			case <-time.NewTimer(interval).C:
			}
			delay = s.fetchAndRecordCloudMetrics(ctx)
		}
	}()
}

// --- Cloud balance (/api/balance) ---

// balancePeriod is the included-credits reset period (non-legacy plans).
type balancePeriod struct {
	From  time.Time `json:"from"`
	Until time.Time `json:"until"`
}

// legacyBalanceLimit is a session or weekly limit (legacy plans).
type legacyBalanceLimit struct {
	RemainingPercent float64   `json:"remaining_percent"`
	ResetsAt         time.Time `json:"resets_at"`
}

// includedBalance is the "included" portion of a balance response. For
// non-legacy plans it has BalanceUSD/AllowanceUSD/Period; for legacy plans
// it has Session/Weekly. The two variants are mutually exclusive.
type includedBalance struct {
	BalanceUSD   float64             `json:"balance_usd,omitempty"`
	AllowanceUSD float64             `json:"allowance_usd,omitempty"`
	Period       *balancePeriod      `json:"period,omitempty"`
	Session      *legacyBalanceLimit `json:"session,omitempty"`
	Weekly       *legacyBalanceLimit `json:"weekly,omitempty"`
}

// purchasedBalance is the purchased-credits portion of a balance response.
type purchasedBalance struct {
	BalanceUSD float64 `json:"balance_usd"`
}

// cloudBalanceResponse is the JSON returned by /api/balance.
type cloudBalanceResponse struct {
	Included  includedBalance  `json:"included"`
	Purchased purchasedBalance `json:"purchased"`
}

// isLegacy returns true when the response uses session/weekly limits
// instead of USD balances.
func (b *cloudBalanceResponse) isLegacy() bool {
	return b.Included.Session != nil || b.Included.Weekly != nil
}

// totalBalance returns included + purchased balance in USD (non-legacy).
func (b *cloudBalanceResponse) totalBalance() float64 {
	return b.Included.BalanceUSD + b.Purchased.BalanceUSD
}

// fetchCloudBalance fetches /api/balance from the cloud service.
func fetchCloudBalance(ctx context.Context) (*cloudBalanceResponse, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	baseURL, err := url.Parse(cloudProxyBaseURL)
	if err != nil {
		return nil, err
	}
	targetURL := baseURL.ResolveReference(&url.URL{Path: "/api/balance"})

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, targetURL.String(), nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	if clientVersion := strings.TrimSpace(version.Version); clientVersion != "" {
		req.Header.Set(cloudProxyClientVersionHeader, clientVersion)
	}

	if err := cloudProxySignRequest(req.Context(), req); err != nil {
		return nil, err
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		retryAfter := 60 * time.Second
		if ra := resp.Header.Get("Retry-After"); ra != "" {
			if secs, err := strconv.Atoi(ra); err == nil && secs > 0 {
				retryAfter = time.Duration(secs) * time.Second
			}
		}
		return nil, &cloudRateLimitError{RetryAfter: retryAfter}
	}

	if resp.StatusCode >= http.StatusBadRequest {
		return nil, fmt.Errorf("cloud balance fetch failed: %s", resp.Status)
	}

	var balance cloudBalanceResponse
	if err := json.NewDecoder(resp.Body).Decode(&balance); err != nil {
		return nil, err
	}
	return &balance, nil
}

// balanceTracker maintains the previous balance snapshot and pending
// per-model request counts to compute per-model usage deltas between
// balance fetches.
//
// For non-legacy plans, the total USD balance delta is attributed to
// models as dollar cost. For legacy plans, the session and weekly
// remaining_percent deltas are attributed as percentage-point usage
// consumed. In both cases, deltas are attributed proportionally by
// request count. When a balance value increases (period reset or
// refund), the corresponding accumulated per-model totals are reset.
type balanceTracker struct {
	mu sync.Mutex
	// Non-legacy USD balance tracking.
	prevTotal  float64
	hasPrev    bool
	modelCosts map[string]float64
	// Legacy session/weekly percentage tracking.
	prevSessionPct    float64
	prevWeeklyPct     float64
	hasPrevLegacy     bool
	modelSessionUsage map[string]float64
	modelWeeklyUsage  map[string]float64
	// Shared: request counts since last balance fetch.
	pendingRequests map[string]int
	// backoffUntil is the time before which balance fetches should be
	// skipped due to a prior 429 response.
	backoffUntil time.Time
}

func newBalanceTracker() *balanceTracker {
	return &balanceTracker{
		modelCosts:        make(map[string]float64),
		modelSessionUsage: make(map[string]float64),
		modelWeeklyUsage:  make(map[string]float64),
		pendingRequests:   make(map[string]int),
	}
}

// recordPendingRequest increments the pending request count for a model.
// Called after each cloud-proxied inference request completes.
func (t *balanceTracker) recordPendingRequest(model string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.pendingRequests[model]++
}

// update computes per-model costs from a new balance snapshot. The total
// balance delta (prevTotal - currentTotal) is attributed to models with
// pending requests, proportionally by request count. Returns a copy of
// the accumulated per-model costs.
func (t *balanceTracker) update(currentTotal float64) map[string]float64 {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.hasPrev {
		t.hasPrev = true
		t.prevTotal = currentTotal
		t.pendingRequests = make(map[string]int)
		return copyMap(t.modelCosts)
	}

	// Balance went up: period reset or refund. Reset accumulated costs.
	if currentTotal > t.prevTotal {
		t.modelCosts = make(map[string]float64)
		t.prevTotal = currentTotal
		t.pendingRequests = make(map[string]int)
		return copyMap(t.modelCosts)
	}

	cost := t.prevTotal - currentTotal
	if cost > 0 && len(t.pendingRequests) > 0 {
		totalPending := 0
		for _, count := range t.pendingRequests {
			totalPending += count
		}
		if totalPending > 0 {
			if len(t.pendingRequests) == 1 {
				for model := range t.pendingRequests {
					t.modelCosts[model] += cost
				}
			} else {
				for model, count := range t.pendingRequests {
					share := float64(count) / float64(totalPending)
					t.modelCosts[model] += cost * share
				}
			}
		}
	}

	t.prevTotal = currentTotal
	t.pendingRequests = make(map[string]int)

	return copyMap(t.modelCosts)
}

// updateLegacy computes per-model session and weekly usage consumption
// from a new legacy balance snapshot. The session and weekly
// remaining_percent deltas (prevPct - currentPct) are attributed to
// models with pending requests, proportionally by request count. When
// a percentage increases (period reset), the corresponding accumulated
// per-model usage is reset. Returns copies of the accumulated per-model
// session and weekly usage maps.
func (t *balanceTracker) updateLegacy(sessionPct, weeklyPct float64) (sessionUsage, weeklyUsage map[string]float64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.hasPrevLegacy {
		t.hasPrevLegacy = true
		t.prevSessionPct = sessionPct
		t.prevWeeklyPct = weeklyPct
		t.pendingRequests = make(map[string]int)
		return copyMap(t.modelSessionUsage), copyMap(t.modelWeeklyUsage)
	}

	// Compute deltas (positive = usage consumed).
	sessionDelta := 0.0
	if sessionPct <= t.prevSessionPct {
		sessionDelta = t.prevSessionPct - sessionPct
	} else {
		// Session reset: percentage went up.
		t.modelSessionUsage = make(map[string]float64)
	}

	weeklyDelta := 0.0
	if weeklyPct <= t.prevWeeklyPct {
		weeklyDelta = t.prevWeeklyPct - weeklyPct
	} else {
		// Weekly reset: percentage went up.
		t.modelWeeklyUsage = make(map[string]float64)
	}

	// Attribute deltas to models proportionally by pending request count.
	if (sessionDelta > 0 || weeklyDelta > 0) && len(t.pendingRequests) > 0 {
		totalPending := 0
		for _, count := range t.pendingRequests {
			totalPending += count
		}
		if totalPending > 0 {
			for model, count := range t.pendingRequests {
				share := float64(count) / float64(totalPending)
				if sessionDelta > 0 {
					t.modelSessionUsage[model] += sessionDelta * share
				}
				if weeklyDelta > 0 {
					t.modelWeeklyUsage[model] += weeklyDelta * share
				}
			}
		}
	}

	t.prevSessionPct = sessionPct
	t.prevWeeklyPct = weeklyPct
	t.pendingRequests = make(map[string]int)

	return copyMap(t.modelSessionUsage), copyMap(t.modelWeeklyUsage)
}

// isRateLimited returns true if the balance API is still in a 429 backoff period.
func (t *balanceTracker) isRateLimited() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return time.Now().Before(t.backoffUntil)
}

// setBackoff records the 429 retry-after duration so subsequent fetches
// are skipped until the backoff expires.
func (t *balanceTracker) setBackoff(d time.Duration) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.backoffUntil = time.Now().Add(d)
}

func copyMap(m map[string]float64) map[string]float64 {
	out := make(map[string]float64, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// recordCloudBalance updates prometheus gauges from a /api/balance response.
func (s *Server) recordCloudBalance(balance *cloudBalanceResponse) {
	if balance == nil || s.sched == nil || s.sched.metrics == nil {
		return
	}

	if balance.isLegacy() {
		if balance.Included.Session != nil {
			s.sched.metrics.cloudSessionRemaining.Set(balance.Included.Session.RemainingPercent)
		}
		if balance.Included.Weekly != nil {
			s.sched.metrics.cloudWeeklyRemaining.Set(balance.Included.Weekly.RemainingPercent)
		}
		s.sched.metrics.cloudPurchasedBalance.Set(balance.Purchased.BalanceUSD)
	} else {
		s.sched.metrics.cloudIncludedBalance.Set(balance.Included.BalanceUSD)
		s.sched.metrics.cloudIncludedAllowance.Set(balance.Included.AllowanceUSD)
		s.sched.metrics.cloudPurchasedBalance.Set(balance.Purchased.BalanceUSD)
	}

	// Compute per-model usage from balance deltas.
	if s.balanceTracker != nil {
		if balance.isLegacy() {
			var sessionPct, weeklyPct float64
			if balance.Included.Session != nil {
				sessionPct = balance.Included.Session.RemainingPercent
			}
			if balance.Included.Weekly != nil {
				weeklyPct = balance.Included.Weekly.RemainingPercent
			}
			sessionUsage, weeklyUsage := s.balanceTracker.updateLegacy(sessionPct, weeklyPct)
			for model, usage := range sessionUsage {
				s.sched.metrics.cloudSessionUsageConsumed.WithLabelValues(model).Set(usage)
			}
			for model, usage := range weeklyUsage {
				s.sched.metrics.cloudWeeklyUsageConsumed.WithLabelValues(model).Set(usage)
			}
		} else {
			costs := s.balanceTracker.update(balance.totalBalance())
			for model, cost := range costs {
				s.sched.metrics.cloudBalanceCost.WithLabelValues(model).Set(cost)
			}
		}
	}
}

// fetchAndRecordCloudBalance fetches /api/balance and records the gauges.
// Returns the rate-limit retry duration if the API returned 429.
func (s *Server) fetchAndRecordCloudBalance(ctx context.Context) time.Duration {
	if s.balanceTracker != nil && s.balanceTracker.isRateLimited() {
		return 0
	}
	balance, err := fetchCloudBalance(ctx)
	if err != nil {
		var rlErr *cloudRateLimitError
		if errors.As(err, &rlErr) {
			slog.Debug("cloud balance rate limited", "retry_after", rlErr.RetryAfter)
			if s.balanceTracker != nil {
				s.balanceTracker.setBackoff(rlErr.RetryAfter)
			}
			return rlErr.RetryAfter
		}
		slog.Debug("cloud balance fetch failed", "error", err)
		return 0
	}
	s.recordCloudBalance(balance)
	return 0
}

// fetchAndRecordCloudMetrics fetches both usage and balance from the cloud
// service and records all metrics. Returns the maximum rate-limit retry
// duration from either fetch.
func (s *Server) fetchAndRecordCloudMetrics(ctx context.Context) time.Duration {
	usageDelay := s.fetchAndRecordCloudUsage(ctx)
	balanceDelay := s.fetchAndRecordCloudBalance(ctx)
	if balanceDelay > usageDelay {
		return balanceDelay
	}
	return usageDelay
}

// cloudProxyMetricsCtx holds mutable state that is populated as
// proxyCloudRequestWithPath progresses and read by the deferred
// metrics recording call. Using a struct avoids passing many
// pointer parameters.
type cloudProxyMetricsCtx struct {
	metricsCapturer   *cloudMetricsCapturer
	cloudBuildCommit  string
	cloudBuildTime    string
	cloudRequestStart time.Time
}

// recordCloudProxyMetrics records all metrics for a cloud-proxied
// request. It is called via defer from proxyCloudRequestWithPath
// so that it runs after the response has been fully written.
func (s *Server) recordCloudProxyMetrics(c *gin.Context, path string, mctx *cloudProxyMetricsCtx) {
	auth := metricsAuthToken(c)
	action := routeToAction(path)
	model, _ := c.Get("cloudModel")
	modelName, _ := model.(string)
	statusCode := c.Writer.Status()
	if statusCode == 0 {
		statusCode = http.StatusInternalServerError
	}
	if statusCode >= 400 {
		s.sched.recordRequestError(action, strings.ToLower(modelName), http.StatusText(statusCode), auth)
	}
	if modelName != "" {
		s.sched.recordCloudRequest(action, strings.ToLower(modelName), auth)
		if mctx.cloudBuildCommit != "" {
			s.sched.recordCloudBuildInfo(strings.ToLower(modelName), mctx.cloudBuildCommit, mctx.cloudBuildTime)
		}
		if s.balanceTracker != nil && statusCode < 400 {
			s.balanceTracker.recordPendingRequest(strings.ToLower(modelName))
		}
	}
	// Record per-model inference metrics captured from the response stream
	if mctx.metricsCapturer != nil && statusCode < 400 {
		if m := mctx.metricsCapturer.metrics(); m != nil {
			m.approximateDurations(mctx.metricsCapturer.ttfb(), time.Since(mctx.cloudRequestStart))
			s.sched.recordPromptAndEvalMetricsWithMemory(strings.ToLower(modelName), m.doneReason, true, m.totalDuration, 0, m.promptEvalDuration, m.evalDuration, m.promptEvalCount, m.evalCount, 0, auth, m.promptEvalCachedCount)
			if s.usageTracker != nil {
				s.usageTracker.recordTokens(strings.ToLower(modelName), m.promptEvalCount, m.promptEvalCachedCount, m.evalCount)
			}
		}
		s.sched.recordTTFT(strings.ToLower(modelName), auth, mctx.metricsCapturer.ttfb())
	}
	// On-demand cloud metrics fetch (when OLLAMA_METRICS_CLOUD_USAGE=on-demand).
	if statusCode < 400 && envconfig.MetricsCloudUsage() == "on-demand" {
		s.fetchAndRecordCloudMetrics(c.Request.Context())
	}
}

type cloudMetricsCapturer struct {
	http.ResponseWriter
	lastLine     []byte
	requestStart time.Time
	ttfbVal      time.Duration
	ttfbOnce     sync.Once
}

func newCloudMetricsCapturer(w http.ResponseWriter, requestStart time.Time) *cloudMetricsCapturer {
	return &cloudMetricsCapturer{ResponseWriter: w, requestStart: requestStart}
}

func (w *cloudMetricsCapturer) ttfb() time.Duration {
	return w.ttfbVal
}

func (w *cloudMetricsCapturer) Write(b []byte) (int, error) {
	if len(b) > 0 {
		w.ttfbOnce.Do(func() { w.ttfbVal = time.Since(w.requestStart) })
	}
	n, err := w.ResponseWriter.Write(b)
	if err != nil {
		return n, err
	}
	// Keep the last non-empty line that looks like it could contain metrics.
	// For Ollama format, the final line has "done": true. For OpenAI format,
	// the final chunk has a "usage" object. Anthropic streaming puts usage in
	// a "message_delta" event, not the last line, so we keep any line with
	// "usage" and prefer it over non-usage lines.
	trimmed := bytes.TrimRight(b, "\n")
	if len(trimmed) > 0 {
		lines := bytes.Split(trimmed, []byte("\n"))
		var fallback []byte
		for i := len(lines) - 1; i >= 0; i-- {
			line := bytes.TrimSpace(lines[i])
			if len(line) == 0 {
				continue
			}
			// Strip SSE "data: " prefix
			content := bytes.TrimPrefix(line, []byte("data: "))
			// Skip [DONE] sentinel
			if bytes.Equal(content, []byte("[DONE]")) {
				continue
			}
			if fallback == nil {
				fallback = line
			}
			// Prefer lines containing usage or done fields
			if bytes.Contains(content, []byte(`"usage"`)) || bytes.Contains(content, []byte(`"done"`)) {
				w.lastLine = line
				break
			}
		}
		if w.lastLine == nil && fallback != nil {
			w.lastLine = fallback
		}
	}
	return n, err
}

func (w *cloudMetricsCapturer) Flush() {
	if flusher, ok := w.ResponseWriter.(http.Flusher); ok {
		flusher.Flush()
	}
}

// cloudCapturedMetrics holds the metrics parsed from the final response.
type cloudCapturedMetrics struct {
	totalDuration         time.Duration
	evalDuration          time.Duration
	promptEvalDuration    time.Duration
	promptEvalCount       int
	promptEvalCachedCount int
	evalCount             int
	doneReason            string
}

// approximateDurations fills in eval and prompt-eval durations from
// proxy-measured timing when the cloud response doesn't include them.
// TTFB approximates prompt eval time; remaining wall time approximates
// eval (generation) time. Assumes zero network latency.
func (m *cloudCapturedMetrics) approximateDurations(ttfb, totalWallTime time.Duration) {
	if m.evalDuration == 0 && totalWallTime > ttfb {
		m.evalDuration = totalWallTime - ttfb
	}
	if m.promptEvalDuration == 0 && ttfb > 0 {
		m.promptEvalDuration = ttfb
	}
}

func (w *cloudMetricsCapturer) metrics() *cloudCapturedMetrics {
	if len(w.lastLine) == 0 {
		return nil
	}
	line := w.lastLine
	// Strip SSE "data: " prefix for OpenAI streaming responses
	line = bytes.TrimPrefix(line, []byte("data: "))
	// Skip the [DONE] sentinel
	if bytes.Equal(line, []byte("[DONE]")) {
		return nil
	}

	// Try Ollama format: {done: true, prompt_eval_count, prompt_eval_cached_count, eval_count, ...}
	var ollamaResp struct {
		Done                  bool   `json:"done"`
		DoneReason            string `json:"done_reason"`
		TotalDuration         int64  `json:"total_duration"`
		EvalDuration          int64  `json:"eval_duration"`
		PromptEvalDuration    int64  `json:"prompt_eval_duration"`
		PromptEvalCount       int    `json:"prompt_eval_count"`
		PromptEvalCachedCount int    `json:"prompt_eval_cached_count"`
		EvalCount             int    `json:"eval_count"`
	}
	if err := json.Unmarshal(line, &ollamaResp); err == nil && ollamaResp.Done {
		return &cloudCapturedMetrics{
			totalDuration:         time.Duration(ollamaResp.TotalDuration),
			evalDuration:          time.Duration(ollamaResp.EvalDuration),
			promptEvalDuration:    time.Duration(ollamaResp.PromptEvalDuration),
			promptEvalCount:       ollamaResp.PromptEvalCount,
			promptEvalCachedCount: ollamaResp.PromptEvalCachedCount,
			evalCount:             ollamaResp.EvalCount,
			doneReason:            ollamaResp.DoneReason,
		}
	}

	// Try OpenAI Chat/Completions format: {usage: {prompt_tokens, completion_tokens, prompt_tokens_details: {cached_tokens}}, ...}
	var openaiResp struct {
		Usage struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(line, &openaiResp); err == nil && (openaiResp.Usage.PromptTokens > 0 || openaiResp.Usage.CompletionTokens > 0) {
		return &cloudCapturedMetrics{
			promptEvalCount:       openaiResp.Usage.PromptTokens,
			promptEvalCachedCount: openaiResp.Usage.PromptTokensDetails.CachedTokens,
			evalCount:             openaiResp.Usage.CompletionTokens,
			doneReason:            "stop",
		}
	}

	// Try OpenAI Responses format: {usage: {input_tokens, output_tokens, input_tokens_details: {cached_tokens}}, ...}
	// or nested {response: {usage: {input_tokens, output_tokens, ...}, ...}} for streaming
	var responsesResp struct {
		Usage struct {
			InputTokens        int `json:"input_tokens"`
			OutputTokens       int `json:"output_tokens"`
			InputTokensDetails struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"input_tokens_details"`
		} `json:"usage"`
		Response struct {
			Usage struct {
				InputTokens        int `json:"input_tokens"`
				OutputTokens       int `json:"output_tokens"`
				InputTokensDetails struct {
					CachedTokens int `json:"cached_tokens"`
				} `json:"input_tokens_details"`
			} `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(line, &responsesResp); err == nil {
		if responsesResp.Usage.InputTokens > 0 || responsesResp.Usage.OutputTokens > 0 {
			return &cloudCapturedMetrics{
				promptEvalCount:       responsesResp.Usage.InputTokens,
				promptEvalCachedCount: responsesResp.Usage.InputTokensDetails.CachedTokens,
				evalCount:             responsesResp.Usage.OutputTokens,
				doneReason:            "stop",
			}
		}
		if responsesResp.Response.Usage.InputTokens > 0 || responsesResp.Response.Usage.OutputTokens > 0 {
			return &cloudCapturedMetrics{
				promptEvalCount:       responsesResp.Response.Usage.InputTokens,
				promptEvalCachedCount: responsesResp.Response.Usage.InputTokensDetails.CachedTokens,
				evalCount:             responsesResp.Response.Usage.OutputTokens,
				doneReason:            "stop",
			}
		}
	}

	return nil
}
