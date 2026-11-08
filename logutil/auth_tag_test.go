package logutil

import (
	"bytes"
	"context"
	"log/slog"
	"strings"
	"testing"
)

func TestRequestIDInLog(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)
	slog.SetDefault(logger)

	ctx := WithRequestID(context.Background(), "test-uuid-123")
	slog.InfoContext(ctx, "test message", "key", "value")

	output := buf.String()
	if !strings.Contains(output, "req_id=test-uuid-123") {
		t.Fatalf("expected req_id in output, got: %s", output)
	}
}

func TestRequestIDNoContext(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)
	slog.SetDefault(logger)

	slog.Info("no context message")
	output := buf.String()
	if strings.Contains(output, "req_id=") {
		t.Fatalf("expected no req_id without context, got: %s", output)
	}
}

func TestRequestIDEmpty(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)
	slog.SetDefault(logger)

	ctx := WithRequestID(context.Background(), "")
	slog.InfoContext(ctx, "empty id message")
	output := buf.String()
	if strings.Contains(output, "req_id=") {
		t.Fatalf("expected no req_id for empty string, got: %s", output)
	}
}

func TestRequestIDPlainContext(t *testing.T) {
	var buf bytes.Buffer
	logger := NewLogger(&buf, slog.LevelDebug)
	slog.SetDefault(logger)

	slog.InfoContext(context.Background(), "plain context message")
	output := buf.String()
	if strings.Contains(output, "req_id=") {
		t.Fatalf("expected no req_id for plain context, got: %s", output)
	}
}

func TestRequestIDFromContext(t *testing.T) {
	ctx := WithRequestID(context.Background(), "abc-123")
	if got := RequestIDFromContext(ctx); got != "abc-123" {
		t.Fatalf("expected abc-123, got %q", got)
	}
	if got := RequestIDFromContext(context.Background()); got != "" {
		t.Fatalf("expected empty string, got %q", got)
	}
}
