package server

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
)

func nowPlus(seconds int) time.Time {
	return time.Now().Add(time.Duration(seconds) * time.Second)
}

func TestNewApiKeyValidator_Unset(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "")
	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v != nil {
		t.Fatal("expected nil validator when OLLAMA_API_KEY is unset")
	}
}

func TestNewApiKeyValidator_List(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "key1 key2 key3")
	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v == nil {
		t.Fatal("expected non-nil validator")
	}
	if got := v.apiKeysForTesting(); got != 3 {
		t.Fatalf("expected 3 keys, got %d", got)
	}
	if !v.valid("key1") {
		t.Fatal("key1 should be valid")
	}
	if !v.valid("key2") {
		t.Fatal("key2 should be valid")
	}
	if !v.valid("key3") {
		t.Fatal("key3 should be valid")
	}
	if v.valid("wrong") {
		t.Fatal("wrong key should be invalid")
	}
}

func TestNewApiKeyValidator_SingleKey(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "mysecret")
	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v == nil {
		t.Fatal("expected non-nil validator")
	}
	if !v.valid("mysecret") {
		t.Fatal("mysecret should be valid")
	}
	if v.valid("nope") {
		t.Fatal("nope should be invalid")
	}
}

func TestNewApiKeyValidator_File(t *testing.T) {
	dir := t.TempDir()
	keyFile := filepath.Join(dir, "keys.txt")
	if err := os.WriteFile(keyFile, []byte("filekey1\nfilekey2\n# comment\n\nfilekey3\n"), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}
	t.Setenv("OLLAMA_API_KEY", keyFile)

	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v == nil {
		t.Fatal("expected non-nil validator")
	}
	if got := v.apiKeysForTesting(); got != 3 {
		t.Fatalf("expected 3 keys (comment and blank lines skipped), got %d", got)
	}
	if !v.valid("filekey1") {
		t.Fatal("filekey1 should be valid")
	}
	if !v.valid("filekey2") {
		t.Fatal("filekey2 should be valid")
	}
	if !v.valid("filekey3") {
		t.Fatal("filekey3 should be valid")
	}
	if v.valid("filekey4") {
		t.Fatal("filekey4 should be invalid")
	}
}

func TestApiKeyValidator_FileReload(t *testing.T) {
	dir := t.TempDir()
	keyFile := filepath.Join(dir, "keys.txt")
	if err := os.WriteFile(keyFile, []byte("oldkey\n"), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}
	t.Setenv("OLLAMA_API_KEY", keyFile)

	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !v.valid("oldkey") {
		t.Fatal("oldkey should be valid")
	}
	if v.valid("newkey") {
		t.Fatal("newkey should be invalid before reload")
	}

	// Update the file — need to change mtime so the reload logic detects it.
	if err := os.WriteFile(keyFile, []byte("oldkey\nnewkey\n"), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}
	// Ensure mtime advances (some filesystems have coarse mtime resolution).
	if err := os.Chtimes(keyFile, nowPlus(2), nowPlus(2)); err != nil {
		t.Fatalf("chtimes: %v", err)
	}

	if !v.valid("oldkey") {
		t.Fatal("oldkey should still be valid")
	}
	if !v.valid("newkey") {
		t.Fatal("newkey should be valid after reload")
	}
}

func TestBearerToken(t *testing.T) {
	tests := []struct {
		header string
		want   string
	}{
		{"", ""},
		{"Bearer mytoken", "mytoken"},
		{"bearer mytoken", "mytoken"},
		{"BEARER mytoken", "mytoken"},
		{"mytoken", "mytoken"},
		{"Bearer  multi-space", "multi-space"},
		{"Bearer", ""},
	}
	for _, tt := range tests {
		got := bearerToken(tt.header)
		if got != tt.want {
			t.Errorf("bearerToken(%q) = %q, want %q", tt.header, got, tt.want)
		}
	}
}

func TestParseKeyToken(t *testing.T) {
	tests := []struct {
		token    string
		wantKey  string
		wantPats []string
	}{
		{"plainkey", "plainkey", nil},
		{"key:/api/generate,/api/chat", "key", []string{"/api/generate", "/api/chat"}},
		{"key:/api/*,!/api/drain", "key", []string{"/api/*", "!/api/drain"}},
		{"key:", "key", nil},
		{"k1:/api/generate", "k1", []string{"/api/generate"}},
	}
	for _, tt := range tests {
		got := parseKeyToken(tt.token)
		if got.key != tt.wantKey {
			t.Errorf("parseKeyToken(%q).key = %q, want %q", tt.token, got.key, tt.wantKey)
		}
		if !slicesEqual(got.patterns, tt.wantPats) {
			t.Errorf("parseKeyToken(%q).patterns = %v, want %v", tt.token, got.patterns, tt.wantPats)
		}
	}
}

func slicesEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestParseKeyLine(t *testing.T) {
	tests := []struct {
		line     string
		wantKey  string
		wantPats []string
	}{
		// Bare key — no patterns, full access
		{"plainkey", "plainkey", nil},

		// Space-separated format
		{"key /api/generate /api/chat", "key", []string{"/api/generate", "/api/chat"}},
		{"key /api/* !/api/drain !/api/delete", "key", []string{"/api/*", "!/api/drain", "!/api/delete"}},

		// Colon/comma format
		{"key:/api/generate,/api/chat", "key", []string{"/api/generate", "/api/chat"}},
		{"key:/api/*,!/api/drail,!/api/delete", "key", []string{"/api/*", "!/api/drail", "!/api/delete"}},

		// Colon format takes precedence when colon appears before any space.
		// Patterns are comma-separated, so spaces within a pattern are preserved.
		{"key:/api/generate,/api/chat extra", "key", []string{"/api/generate", "/api/chat extra"}},

		// Whitespace-only line
		{"   ", "", nil},
	}
	for _, tt := range tests {
		got := parseKeyLine(tt.line)
		if got.key != tt.wantKey {
			t.Errorf("parseKeyLine(%q).key = %q, want %q", tt.line, got.key, tt.wantKey)
		}
		if !slicesEqual(got.patterns, tt.wantPats) {
			t.Errorf("parseKeyLine(%q).patterns = %v, want %v", tt.line, got.patterns, tt.wantPats)
		}
	}
}

func TestPathAllowed(t *testing.T) {
	tests := []struct {
		name     string
		path     string
		patterns []string
		want     bool
	}{
		{"no patterns = full access", "/api/drain", nil, true},

		// Allow rules
		{"exact match", "/api/generate", []string{"/api/generate"}, true},
		{"no match", "/api/drain", []string{"/api/generate"}, false},
		{"glob match", "/api/generate", []string{"/api/*"}, true},
		{"glob no match", "/v1/chat", []string{"/api/*"}, false},

		// Deny rules
		{"deny takes precedence", "/api/drain", []string{"/api/*", "!/api/drain"}, false},
		{"allow after deny check passes", "/api/generate", []string{"/api/*", "!/api/drain"}, true},
		{"deny only", "/api/delete", []string{"!/api/delete"}, false},
		{"deny glob", "/api/drain", []string{"!/api/d*"}, false},

		// Multiple patterns
		{"multi allow", "/api/chat", []string{"/api/generate", "/api/chat"}, true},
		{"multi allow no match", "/api/embed", []string{"/api/generate", "/api/chat"}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := pathAllowed(tt.path, tt.patterns); got != tt.want {
				t.Errorf("pathAllowed(%q, %v) = %v, want %v", tt.path, tt.patterns, got, tt.want)
			}
		})
	}
}

func TestEnvVarWithPatterns(t *testing.T) {
	t.Setenv("OLLAMA_API_KEY", "plain-key admin-key:/api/*,!/api/drain,!/api/delete readonly-key:/api/generate,/api/chat")
	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v == nil {
		t.Fatal("expected non-nil validator")
	}
	if got := v.apiKeysForTesting(); got != 3 {
		t.Fatalf("expected 3 keys, got %d", got)
	}

	// plain-key: full access
	matched, allowed := v.authorize("plain-key", "/api/drain")
	if !matched || !allowed {
		t.Fatal("plain-key should have full access")
	}

	// admin-key: allowed except drain/delete
	matched, allowed = v.authorize("admin-key", "/api/generate")
	if !matched || !allowed {
		t.Fatal("admin-key should be allowed for /api/generate")
	}
	matched, allowed = v.authorize("admin-key", "/api/drain")
	if !matched || allowed {
		t.Fatal("admin-key should be denied for /api/drain")
	}

	// readonly-key: only generate/chat
	matched, allowed = v.authorize("readonly-key", "/api/chat")
	if !matched || !allowed {
		t.Fatal("readonly-key should be allowed for /api/chat")
	}
	matched, allowed = v.authorize("readonly-key", "/api/drain")
	if !matched || allowed {
		t.Fatal("readonly-key should be denied for /api/drain")
	}

	// wrong key
	matched, _ = v.authorize("wrong", "/api/generate")
	if matched {
		t.Fatal("wrong key should not match")
	}
}

func TestFileWithPatterns(t *testing.T) {
	dir := t.TempDir()
	keyFile := filepath.Join(dir, "keys.txt")
	content := "plain-key\n" +
		"admin-key  /api/*  !/api/drain  !/api/delete\n" +
		"readonly-key:/api/generate,/api/chat\n"
	if err := os.WriteFile(keyFile, []byte(content), 0o600); err != nil {
		t.Fatalf("write key file: %v", err)
	}
	t.Setenv("OLLAMA_API_KEY", keyFile)

	v, err := newApiKeyValidator()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := v.apiKeysForTesting(); got != 3 {
		t.Fatalf("expected 3 keys, got %d", got)
	}

	// plain-key: full access
	matched, allowed := v.authorize("plain-key", "/api/drain")
	if !matched || !allowed {
		t.Fatal("plain-key should have full access")
	}

	// admin-key (space format): allowed except drain/delete
	matched, allowed = v.authorize("admin-key", "/api/generate")
	if !matched || !allowed {
		t.Fatal("admin-key should be allowed for /api/generate")
	}
	matched, allowed = v.authorize("admin-key", "/api/drain")
	if !matched || allowed {
		t.Fatal("admin-key should be denied for /api/drain")
	}

	// readonly-key (colon format): only generate/chat
	matched, allowed = v.authorize("readonly-key", "/api/chat")
	if !matched || !allowed {
		t.Fatal("readonly-key should be allowed for /api/chat")
	}
	matched, allowed = v.authorize("readonly-key", "/api/drain")
	if !matched || allowed {
		t.Fatal("readonly-key should be denied for /api/drain")
	}
}

func TestApiKeyMiddleware(t *testing.T) {
	gin.SetMode(gin.TestMode)

	tests := []struct {
		name       string
		validator  *apiKeyValidator
		authHeader string
		path       string
		wantStatus int
	}{
		{
			name:       "no validator, no auth header -> allowed",
			validator:  nil,
			authHeader: "",
			path:       "/",
			wantStatus: http.StatusOK,
		},
		{
			name:       "no validator, auth header -> allowed",
			validator:  nil,
			authHeader: "Bearer anything",
			path:       "/",
			wantStatus: http.StatusOK,
		},
		{
			name:       "validator set, matching key -> allowed",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret"}}},
			authHeader: "Bearer secret",
			path:       "/api/generate",
			wantStatus: http.StatusOK,
		},
		{
			name:       "validator set, non-matching key -> 401",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret"}}},
			authHeader: "Bearer wrong",
			path:       "/api/generate",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "validator set, no auth header -> 401",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret"}}},
			authHeader: "",
			path:       "/api/generate",
			wantStatus: http.StatusUnauthorized,
		},
		{
			name:       "validator set, raw token without Bearer prefix -> allowed",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret"}}},
			authHeader: "secret",
			path:       "/api/generate",
			wantStatus: http.StatusOK,
		},
		{
			name:       "key with path restriction, allowed path -> 200",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret", patterns: []string{"/api/generate"}}}},
			authHeader: "Bearer secret",
			path:       "/api/generate",
			wantStatus: http.StatusOK,
		},
		{
			name:       "key with path restriction, denied path -> 403",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret", patterns: []string{"/api/generate"}}}},
			authHeader: "Bearer secret",
			path:       "/api/drain",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "key with deny rule, denied path -> 403",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret", patterns: []string{"/api/*", "!/api/drain"}}}},
			authHeader: "Bearer secret",
			path:       "/api/drain",
			wantStatus: http.StatusForbidden,
		},
		{
			name:       "key with deny rule, allowed path -> 200",
			validator:  &apiKeyValidator{entries: []apiKeyEntry{{key: "secret", patterns: []string{"/api/*", "!/api/drain"}}}},
			authHeader: "Bearer secret",
			path:       "/api/generate",
			wantStatus: http.StatusOK,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := gin.New()
			r.Use(apiKeyMiddleware(tt.validator))
			r.GET(tt.path, func(c *gin.Context) { c.Status(http.StatusOK) })

			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			if tt.authHeader != "" {
				req.Header.Set("Authorization", tt.authHeader)
			}
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d", w.Code, tt.wantStatus)
			}
		})
	}
}
