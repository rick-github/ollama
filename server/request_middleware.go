package server

import (
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"log/slog"

	"github.com/ollama/ollama/logutil"
)

// requestIDMiddleware generates a UUID for each request and stores it in the
// request context.  Log records emitted via slog *Context calls on the
// request path will carry the ID as a structured "req_id" field.  The UUID
// is also set as a gin context key so the custom gin logger can include it
// in the request completion line.
func requestIDMiddleware(c *gin.Context) {
	id := uuid.NewString()
	c.Request = c.Request.WithContext(logutil.WithRequestID(c.Request.Context(), id))
	c.Set("request_id", id)
	c.Header("x-request-id", id)
	c.Next()
}

// requestIDLogger is a gin middleware that logs each request at completion
// with the request UUID, replacing gin's default logger.  The format mirrors
// gin's default but appends the request ID.
func requestIDLogger() gin.HandlerFunc {
	return func(c *gin.Context) {
		skipLogPaths := map[string]bool{"/metrics": true}
		if skipLogPaths[c.Request.URL.Path] {
			c.Next()
			return
		}
		start := time.Now()
		c.Next()
		id, _ := c.Get("request_id")
		cloudID, _ := c.Get("cloud_request_id")
		latency := time.Since(start)
		if latency > time.Minute {
			latency = latency.Truncate(time.Second)
		}
		args := []any{
			"req_id", id,
			"status", c.Writer.Status(),
			"latency", latency.String(),
			"client", c.ClientIP(),
			"method", c.Request.Method,
			"path", c.Request.URL.Path,
		}
		if cloudID != nil && cloudID != "" {
			args = append(args, "cloud_req_id", cloudID)
		}
		slog.Info("request completed", args...)
	}
}
