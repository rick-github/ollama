package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/ml"
	"github.com/stretchr/testify/require"
)

func TestHealthHandler_NoModels(t *testing.T) {
	gin.SetMode(gin.TestMode)

	s := &Server{
		sched:       InitScheduler(context.Background()),
		modelCaches: newModelCaches(),
	}

	router, err := s.GenerateRoutes()
	require.NoError(t, err)

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	resp, err := http.Get(httpSrv.URL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	var hr api.HealthResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&hr))
	require.Equal(t, "ok", hr.Status)
	require.Empty(t, hr.Models)
}

func TestHealthHandler_AllHealthy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	sched := InitScheduler(context.Background())

	runner := &runnerRef{
		modelPath: "/fake/model",
		modelKey:  "/fake/model",
		llama:     &mockLlm{vramSize: 1 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{}},
		refCount:  1,
	}
	runner.model.Store(&Model{Name: "test-model", ModelPath: "/fake/model", Digest: "sha256-abc"})

	sched.loadedMu.Lock()
	sched.loaded["/fake/model"] = runner
	sched.loadedMu.Unlock()

	s := &Server{
		sched:       sched,
		modelCaches: newModelCaches(),
	}

	router, err := s.GenerateRoutes()
	require.NoError(t, err)

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	resp, err := http.Get(httpSrv.URL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusOK, resp.StatusCode)

	var hr api.HealthResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&hr))
	require.Equal(t, "ok", hr.Status)
	require.Len(t, hr.Models, 1)
	require.Equal(t, "test-model", hr.Models[0].Name)
	require.Equal(t, "ok", hr.Models[0].Status)
}

func TestHealthHandler_UnhealthyRunner(t *testing.T) {
	gin.SetMode(gin.TestMode)

	sched := InitScheduler(context.Background())

	runner := &runnerRef{
		modelPath: "/fake/bad",
		modelKey:  "/fake/bad",
		llama: &mockLlm{
			vramSize:  1 * format.GigaByte,
			vramByGPU: map[ml.DeviceID]uint64{},
			pingResp:  errors.New("connection refused"),
		},
		refCount: 1,
	}
	runner.model.Store(&Model{Name: "bad-model", ModelPath: "/fake/bad", Digest: "sha256-bad"})

	sched.loadedMu.Lock()
	sched.loaded["/fake/bad"] = runner
	sched.loadedMu.Unlock()

	s := &Server{
		sched:       sched,
		modelCaches: newModelCaches(),
	}

	router, err := s.GenerateRoutes()
	require.NoError(t, err)

	httpSrv := httptest.NewServer(router)
	t.Cleanup(httpSrv.Close)

	resp, err := http.Get(httpSrv.URL + "/health")
	require.NoError(t, err)
	defer resp.Body.Close()

	require.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)

	var hr api.HealthResponse
	require.NoError(t, json.NewDecoder(resp.Body).Decode(&hr))
	require.Equal(t, "unhealthy", hr.Status)
	require.Len(t, hr.Models, 1)
	require.Equal(t, "unhealthy", hr.Models[0].Status)
	require.Contains(t, hr.Models[0].Message, "connection refused")
}

func TestHealthHandler_MixedRunners(t *testing.T) {
	gin.SetMode(gin.TestMode)

	sched := InitScheduler(context.Background())

	healthy := &runnerRef{
		modelPath: "/fake/good",
		modelKey:  "/fake/good",
		llama:     &mockLlm{vramSize: 1 * format.GigaByte, vramByGPU: map[ml.DeviceID]uint64{}},
		refCount:  1,
	}
	healthy.model.Store(&Model{Name: "good", ModelPath: "/fake/good", Digest: "sha256-good"})

	unhealthy := &runnerRef{
		modelPath: "/fake/bad",
		modelKey:  "/fake/bad",
		llama: &mockLlm{
			vramSize:  1 * format.GigaByte,
			vramByGPU: map[ml.DeviceID]uint64{},
			pingResp:  errors.New("timeout"),
		},
		refCount: 1,
	}
	unhealthy.model.Store(&Model{Name: "bad", ModelPath: "/fake/bad", Digest: "sha256-bad"})

	sched.loadedMu.Lock()
	sched.loaded["/fake/good"] = healthy
	sched.loaded["/fake/bad"] = unhealthy
	sched.loadedMu.Unlock()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	results := sched.healthCheck(ctx)
	require.Len(t, results, 2)

	statusMap := map[string]string{}
	for _, r := range results {
		statusMap[r.Name] = r.Status
	}
	require.Equal(t, "ok", statusMap["good"])
	require.Equal(t, "unhealthy", statusMap["bad"])
}
