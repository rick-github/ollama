package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

func TestDrainSetAndClear(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()

	s := InitScheduler(ctx)

	require.False(t, s.IsDraining(), "drain should be off by default")

	s.SetDraining(true)
	require.True(t, s.IsDraining(), "drain should be on after SetDraining(true)")

	s.SetDraining(false)
	require.False(t, s.IsDraining(), "drain should be off after SetDraining(false)")
}

func TestDrainRejectsNewRequests(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()

	s := InitScheduler(ctx)
	successCh, errCh := s.getRunner(ctx, &Model{
		Name:      "test",
		ModelPath: "test",
	}, api.DefaultOptions(), nil, false, false, nil)

	// Without drain, the request should go to pending (no runner loaded, no
	// real model on disk, so it will sit in the queue). We just verify there
	// is no immediate error.
	select {
	case err := <-errCh:
		// If we got ErrMaxQueue the channel buffer was full, which shouldn't
		// happen with a fresh scheduler. But drain isn't set so ErrDraining
		// must not appear.
		require.NotErrorIs(t, err, ErrDraining, "should not get ErrDraining when not draining")
	default:
	}

	// Drain the pending request manually, updating the tracking map to
	// match what processPending would do.
	if req, ok := <-s.pendingReqCh; ok {
		s.pendingMu.Lock()
		s.pendingModels[req.model.Name]--
		if s.pendingModels[req.model.Name] <= 0 {
			delete(s.pendingModels, req.model.Name)
		}
		s.pendingMu.Unlock()
	}

	// Enable drain and verify the next request is rejected immediately
	s.SetDraining(true)
	_ = successCh
	_, errCh2 := s.getRunner(ctx, &Model{
		Name:      "test",
		ModelPath: "test",
	}, api.DefaultOptions(), nil, false, false, nil)

	select {
	case err := <-errCh2:
		require.ErrorIs(t, err, ErrDraining, "should get ErrDraining when draining")
	default:
		t.Fatal("expected ErrDraining on errCh")
	}
}

func TestPendingByModel(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()

	s := InitScheduler(ctx)

	// Queue requests for two different models
	for _, name := range []string{"llama3:8b", "llama3:8b", "qwen2:7b"} {
		_, _ = s.getRunner(ctx, &Model{
			Name:      name,
			ModelPath: name,
		}, api.DefaultOptions(), nil, false, false, nil)
	}

	byModel := s.PendingByModel()
	require.Equal(t, 2, byModel["llama3:8b"])
	require.Equal(t, 1, byModel["qwen2:7b"])
	require.Equal(t, 3, s.PendingQueueDepth())
}

func TestDrainHandler(t *testing.T) {
	ctx, done := context.WithCancel(t.Context())
	defer done()

	sched := InitScheduler(ctx)
	srv := &Server{sched: sched}

	t.Run("POST enables drain", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/drain", nil)
		c.Request.Body = http.NoBody
		// Manually set JSON body
		c.Request.Header.Set("Content-Type", "application/json")
		body := `{"drain": true}`
		c.Request.Body = nopCloser{[]byte(body)}
		c.Request.ContentLength = int64(len(body))

		srv.DrainHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.True(t, resp.Draining)
		require.True(t, sched.IsDraining())
	})

	t.Run("POST disables drain", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/drain", nil)
		c.Request.Header.Set("Content-Type", "application/json")
		body := `{"drain": false}`
		c.Request.Body = nopCloser{[]byte(body)}
		c.Request.ContentLength = int64(len(body))

		srv.DrainHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.False(t, resp.Draining)
		require.False(t, sched.IsDraining())
	})

	t.Run("POST unload only does not change drain", func(t *testing.T) {
		sched.SetDraining(true)
		defer sched.SetDraining(false)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/drain", nil)
		c.Request.Header.Set("Content-Type", "application/json")
		body := `{"unload": true}`
		c.Request.Body = nopCloser{[]byte(body)}
		c.Request.ContentLength = int64(len(body))

		srv.DrainHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.True(t, resp.Draining, "drain should remain on")
		require.True(t, sched.IsDraining(), "drain should remain on")
		require.Zero(t, resp.LoadedModels, "all models should be unloaded")
	})

	t.Run("POST empty body does not change drain", func(t *testing.T) {
		sched.SetDraining(true)
		defer sched.SetDraining(false)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/drain", nil)
		c.Request.Header.Set("Content-Type", "application/json")
		body := `{}`
		c.Request.Body = nopCloser{[]byte(body)}
		c.Request.ContentLength = int64(len(body))

		srv.DrainHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.True(t, resp.Draining, "drain should remain on")
	})

	t.Run("POST graceful unload with no runners returns immediately", func(t *testing.T) {
		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodPost, "/api/drain", nil)
		c.Request.Header.Set("Content-Type", "application/json")
		body := `{"unload": true, "graceful": true}`
		c.Request.Body = nopCloser{[]byte(body)}
		c.Request.ContentLength = int64(len(body))

		srv.DrainHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.Zero(t, resp.LoadedModels, "no models to unload")
	})

	t.Run("GET returns status", func(t *testing.T) {
		sched.SetDraining(true)
		defer sched.SetDraining(false)

		w := httptest.NewRecorder()
		c, _ := gin.CreateTestContext(w)
		c.Request = httptest.NewRequest(http.MethodGet, "/api/drain", nil)

		srv.DrainStatusHandler(c)

		require.Equal(t, http.StatusOK, w.Code)
		var resp api.DrainResponse
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		require.True(t, resp.Draining)
		require.Zero(t, resp.PendingQueue)
		require.Empty(t, resp.PendingQueueByModel)
	})
}

type nopCloser struct {
	data []byte
}

func (n nopCloser) Read(p []byte) (int, error) {
	if len(n.data) == 0 {
		return 0, http.ErrBodyReadAfterClose
	}
	copied := copy(p, n.data)
	n.data = n.data[copied:]
	return copied, nil
}

func (n nopCloser) Close() error { return nil }
