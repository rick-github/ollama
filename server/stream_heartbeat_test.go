package server

import (
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func TestStreamResponseWithHeartbeat(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("sends heartbeat when channel is idle", func(t *testing.T) {
		w := NewRecorder()
		c, _ := gin.CreateTestContext(w)

		ch := make(chan any, 10)

		// Start streaming with a short heartbeat interval
		done := make(chan struct{})
		go func() {
			streamResponseWithHeartbeat(c, ch, 50*time.Millisecond)
			close(done)
		}()

		// Wait long enough for at least 2 heartbeats to fire
		time.Sleep(200 * time.Millisecond)

		// Now send a real response and close
		ch <- gin.H{"message": "hello", "done": true}
		close(ch)

		<-done

		body := w.Body.String()
		lines := splitLines(body)

		// Should have at least 2 heartbeats and 1 real response
		heartbeats := 0
		responses := 0
		for _, line := range lines {
			if line == "" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal([]byte(line), &m); err != nil {
				t.Fatalf("failed to parse line: %s: %v", line, err)
			}
			if _, ok := m["done"]; ok {
				if done, isBool := m["done"].(bool); isBool && !done {
					heartbeats++
				} else {
					responses++
				}
			} else {
				responses++
			}
		}

		if heartbeats < 2 {
			t.Fatalf("expected at least 2 heartbeats, got %d (body: %s)", heartbeats, body)
		}
		if responses != 1 {
			t.Fatalf("expected 1 real response, got %d (body: %s)", responses, body)
		}
	})

	t.Run("no heartbeat when disabled", func(t *testing.T) {
		w := NewRecorder()
		c, _ := gin.CreateTestContext(w)

		ch := make(chan any, 10)

		done := make(chan struct{})
		go func() {
			streamResponseWithHeartbeat(c, ch, 0) // disabled
			close(done)
		}()

		// Wait to prove no heartbeat arrives
		time.Sleep(100 * time.Millisecond)

		ch <- gin.H{"message": "hello", "done": true}
		close(ch)

		<-done

		body := w.Body.String()
		for _, line := range splitLines(body) {
			if line == "" {
				continue
			}
			var m map[string]any
			if err := json.Unmarshal([]byte(line), &m); err != nil {
				t.Fatalf("failed to parse line: %s: %v", line, err)
			}
			if v, ok := m["done"]; ok {
				if done, isBool := v.(bool); isBool && !done {
					t.Fatalf("unexpected heartbeat when disabled: %s", body)
				}
			}
		}
	})
}

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i, c := range s {
		if c == '\n' {
			lines = append(lines, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		lines = append(lines, s[start:])
	}
	return lines
}

var _ = httptest.NewRecorder
