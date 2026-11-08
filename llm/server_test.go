package llm

import (
	"testing"
)

func TestResolveKvCacheType(t *testing.T) {
	tests := []struct {
		name     string
		perModel string
		env      string
		setEnv   bool
		want     string
	}{
		{
			name:   "empty per-model falls back to env",
			env:    "q8_0",
			setEnv: true,
			want:   "q8_0",
		},
		{
			name: "empty per-model and empty env yields empty",
			want: "",
		},
		{
			name:     "per-model overrides env",
			perModel: "q4_0",
			env:      "q8_0",
			setEnv:   true,
			want:     "q4_0",
		},
		{
			name:     "per-model lowercased",
			perModel: "Q8_0",
			want:     "q8_0",
		},
		{
			name:     "per-model whitespace trimmed and lowercased",
			perModel: "  Q4_1  ",
			want:     "q4_1",
		},
		{
			name:     "whitespace-only per-model falls back to env",
			perModel: "   ",
			env:      "f16",
			setEnv:   true,
			want:     "f16",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setEnv {
				t.Setenv("OLLAMA_KV_CACHE_TYPE", tt.env)
			}
			if got := resolveKvCacheType(tt.perModel); got != tt.want {
				t.Fatalf("resolveKvCacheType(%q) = %q, want %q", tt.perModel, got, tt.want)
			}
		})
	}
}
