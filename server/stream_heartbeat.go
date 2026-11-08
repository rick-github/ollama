package server

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// streamResponseWithHeartbeat streams responses from ch as newline-delimited
// JSON. When heartbeat is non-zero, it writes a keepalive response
// ({"done":false}) at that interval when the channel is idle, preventing
// clients with short idle timeouts from aborting the connection. The
// keepalive is a valid ChatResponse/GenerateResponse with no content, so
// existing clients handle it without any special logic.
func streamResponseWithHeartbeat(c *gin.Context, ch chan any, heartbeat time.Duration) {
	c.Header("Content-Type", "application/x-ndjson")

	var heartbeatTicker *time.Ticker
	var heartbeatC <-chan time.Time
	if heartbeat > 0 {
		heartbeatTicker = time.NewTicker(heartbeat)
		defer heartbeatTicker.Stop()
		heartbeatC = heartbeatTicker.C
	}

	c.Stream(func(w io.Writer) bool {
		select {
		case val, ok := <-ch:
			if !ok {
				return false
			}

			// errors are provided as a gin.H with an "error" field and
			// an optional "status" field.  For errors that are streamed
			// before any content, we need to set the status code and
			// content type for the error.
			if h, ok := val.(gin.H); ok {
				if e, ok := h["error"].(string); ok {
					status, ok := h["status"].(int)
					if !ok {
						status = http.StatusInternalServerError
					}

					if !c.Writer.Written() {
						c.Header("Content-Type", "application/json")
						c.JSON(status, gin.H{"error": e})
					} else {
						if err := json.NewEncoder(c.Writer).Encode(gin.H{"error": e}); err != nil {
							slog.ErrorContext(c.Request.Context(), "streamResponse failed to encode json error", "error", err)
						}
					}

					return false
				}
			}

			bts, err := json.Marshal(val)
			if err != nil {
				slog.InfoContext(c.Request.Context(), fmt.Sprintf("streamResponse: json.Marshal failed with %s", err))
				return false
			}

			// Delineate chunks with new-line delimiter
			bts = append(bts, '\n')
			if _, err := w.Write(bts); err != nil {
				slog.InfoContext(c.Request.Context(), fmt.Sprintf("streamResponse: w.Write failed with %s", err))
				return false
			}

			return true
		case <-heartbeatC:
			if _, err := w.Write([]byte(`{"done":false}` + "\n")); err != nil {
				slog.InfoContext(c.Request.Context(), fmt.Sprintf("streamResponse: heartbeat write failed: %s", err))
				return false
			}
			return true
		}
	})
}
