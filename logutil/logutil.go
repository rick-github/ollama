package logutil

import (
	"context"
	"io"
	"log/slog"
	"path/filepath"
	"runtime"
	"time"
)

const LevelTrace slog.Level = -8

func NewLogger(w io.Writer, level slog.Level) *slog.Logger {
	return slog.New(requestHandler{slog.NewTextHandler(w, &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(_ []string, attr slog.Attr) slog.Attr {
			switch attr.Key {
			case slog.LevelKey:
				switch attr.Value.Any().(slog.Level) {
				case LevelTrace:
					attr.Value = slog.StringValue("TRACE")
				}
			case slog.SourceKey:
				if source, ok := attr.Value.Any().(*slog.Source); ok {
					source.File = filepath.Base(source.File)
				}
			}
			return attr
		},
	})})
}

type key string

const reqIDKey key = "req_id"

// WithRequestID stores a request ID (UUID) in the context so that log
// records emitted with that context (via slog.InfoContext, slog.DebugContext,
// etc.) include the ID as a structured "req_id" field.
func WithRequestID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, reqIDKey, id)
}

// RequestIDFromContext extracts the request ID from the context, if present.
func RequestIDFromContext(ctx context.Context) string {
	if id, ok := ctx.Value(reqIDKey).(string); ok {
		return id
	}
	return ""
}

// requestHandler wraps a slog.Handler and injects a "req_id" from the
// context into every record that is handled.
type requestHandler struct {
	slog.Handler
}

func (h requestHandler) Handle(ctx context.Context, record slog.Record) error {
	if id, ok := ctx.Value(reqIDKey).(string); ok && id != "" {
		record.AddAttrs(slog.String("req_id", id))
	}
	return h.Handler.Handle(ctx, record)
}

func Trace(msg string, args ...any) {
	TraceContext(context.WithValue(context.TODO(), key("skip"), 1), msg, args...)
}

func TraceContext(ctx context.Context, msg string, args ...any) {
	if logger := slog.Default(); logger.Enabled(ctx, LevelTrace) {
		skip, _ := ctx.Value(key("skip")).(int)
		pc, _, _, _ := runtime.Caller(1 + skip)
		record := slog.NewRecord(time.Now(), LevelTrace, msg, pc)
		record.Add(args...)
		logger.Handler().Handle(ctx, record)
	}
}
