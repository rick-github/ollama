package server

import (
	"bufio"
	"fmt"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/envconfig"
)

// apiKeyEntry represents a single API key and its optional endpoint
// restrictions. When patterns is empty, the key has unrestricted access.
// Pattern entries starting with "!" are deny rules.
type apiKeyEntry struct {
	key      string
	patterns []string // empty = allow all; entries prefixed with "!" are deny rules
}

// apiKeyValidator checks incoming Authorization headers against a set of
// accepted API keys.  Keys are sourced from the OLLAMA_API_KEY environment
// variable, which may contain either a space-separated list of keys or the
// path to a file with one key per line.  When a file is used, its modification
// time is checked on every validation so that new keys take effect without a
// server restart.
//
// Keys may optionally carry endpoint restrictions.  In the env var, the format
// is "key:pat1,pat2" (colon separates key from comma-separated patterns).  In
// a key file, both "key:pat1,pat2" and "key pat1 pat2" (space-separated) are
// accepted.  A key without patterns has unrestricted access.
type apiKeyValidator struct {
	mu sync.Mutex

	// source is the raw value of OLLAMA_API_KEY at construction time.
	source string

	// isFile is true when source was detected as an existing file path.
	isFile bool

	// entries is the current set of accepted keys and their endpoint patterns.
	entries []apiKeyEntry

	// fileMod is the modification time of the file when it was last loaded.
	fileMod time.Time
}

// newApiKeyValidator creates a validator from the current OLLAMA_API_KEY
// value.  Returns nil (with no error) when the env var is unset or empty,
// indicating that API key validation is disabled.
func newApiKeyValidator() (*apiKeyValidator, error) {
	src := strings.TrimSpace(envconfig.ApiKey())
	if src == "" {
		return nil, nil
	}

	v := &apiKeyValidator{source: src}

	// If the value looks like a path to an existing file, treat it as a file.
	// Otherwise, treat it as a space-separated list of keys.
	if fi, err := os.Stat(src); err == nil && !fi.IsDir() {
		v.isFile = true
		if err := v.reloadFile(); err != nil {
			return nil, fmt.Errorf("OLLAMA_API_KEY: %w", err)
		}
	} else {
		for _, token := range strings.Fields(src) {
			v.entries = append(v.entries, parseKeyToken(token))
		}
		if len(v.entries) == 0 {
			return nil, nil
		}
	}

	return v, nil
}

// parseKeyToken parses a single env-var token.  The format is:
//
//	key                 → full access
//	key:pat1,pat2       → restricted to pat1 and pat2
//
// Patterns are comma-separated.  Entries prefixed with "!" are deny rules.
func parseKeyToken(token string) apiKeyEntry {
	idx := strings.IndexByte(token, ':')
	if idx < 0 {
		return apiKeyEntry{key: token}
	}
	key := token[:idx]
	rest := token[idx+1:]
	var patterns []string
	for _, p := range strings.Split(rest, ",") {
		p = strings.TrimSpace(p)
		if p != "" {
			patterns = append(patterns, p)
		}
	}
	return apiKeyEntry{key: key, patterns: patterns}
}

// parseKeyLine parses a single line from a key file.  Both formats are
// accepted:
//
//	key pat1 pat2         → space-separated patterns
//	key:pat1,pat2         → colon/comma-separated patterns
//	key                   → full access
//
// If the line contains a colon before any whitespace, the colon/comma format
// is used.  Otherwise, the space-separated format is used.
func parseKeyLine(line string) apiKeyEntry {
	line = strings.TrimSpace(line)

	// Detect colon/comma format: colon appears before any whitespace,
	// meaning the key itself doesn't contain spaces (keys are alphanum).
	colonIdx := strings.IndexByte(line, ':')
	spaceIdx := strings.IndexByte(line, ' ')

	if colonIdx >= 0 && (spaceIdx < 0 || colonIdx < spaceIdx) {
		return parseKeyToken(line)
	}

	// Space-separated format.
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return apiKeyEntry{}
	}
	return apiKeyEntry{
		key:      fields[0],
		patterns: fields[1:],
	}
}

// reloadFile re-reads the key file and updates the in-memory entry set.
// The caller must hold v.mu.
func (v *apiKeyValidator) reloadFile() error {
	fi, err := os.Stat(v.source)
	if err != nil {
		return fmt.Errorf("stat key file: %w", err)
	}

	f, err := os.Open(v.source)
	if err != nil {
		return fmt.Errorf("open key file: %w", err)
	}
	defer f.Close()

	var entries []apiKeyEntry
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		entries = append(entries, parseKeyLine(line))
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read key file: %w", err)
	}

	v.entries = entries
	v.fileMod = fi.ModTime()
	return nil
}

// authorize checks whether token is accepted and, if so, whether the request
// path is allowed for that key.  When the keys are sourced from a file, the
// file's modification time is checked first and the file is reloaded if it has
// changed.  Returns (matched, allowed) where matched is false if the token
// doesn't match any key at all.
func (v *apiKeyValidator) authorize(token, reqPath string) (matched, allowed bool) {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.isFile {
		if fi, err := os.Stat(v.source); err == nil && fi.ModTime() != v.fileMod {
			if err := v.reloadFile(); err != nil {
				return v.lookup(token, reqPath)
			}
		}
	}

	return v.lookup(token, reqPath)
}

// lookup finds the entry for token and checks whether reqPath is allowed.
// The caller must hold v.mu.
func (v *apiKeyValidator) lookup(token, reqPath string) (matched, allowed bool) {
	for _, e := range v.entries {
		if e.key == token {
			return true, pathAllowed(reqPath, e.patterns)
		}
	}
	return false, false
}

// pathAllowed checks whether reqPath matches the given patterns.  An empty
// pattern list means allow all.  Entries prefixed with "!" are deny rules
// and take precedence over allow rules.
func pathAllowed(reqPath string, patterns []string) bool {
	if len(patterns) == 0 {
		return true
	}

	// Deny rules take precedence: if any deny rule matches, reject.
	for _, p := range patterns {
		if strings.HasPrefix(p, "!") {
			deny := p[1:]
			if ok, _ := path.Match(deny, reqPath); ok {
				return false
			}
		}
	}

	// Then check allow rules.
	for _, p := range patterns {
		if strings.HasPrefix(p, "!") {
			continue
		}
		if ok, _ := path.Match(p, reqPath); ok {
			return true
		}
	}

	return false
}

// apiKeyMiddleware returns gin middleware that enforces API key authentication
// when OLLAMA_API_KEY is configured.  The middleware runs before route handlers
// and rejects requests that do not supply a matching bearer token.  When a key
// has endpoint restrictions, requests to non-allowed endpoints receive 403.
//
// Behaviour summary:
//
//	OLLAMA_API_KEY set     + valid key + path allowed       → allowed
//	OLLAMA_API_KEY set     + valid key + path not allowed  → 403
//	OLLAMA_API_KEY set     + invalid key                   → 401
//	OLLAMA_API_KEY set     + no Authorization header       → 401
//	OLLAMA_API_KEY unset   + any                           → allowed
func apiKeyMiddleware(validator *apiKeyValidator) gin.HandlerFunc {
	return func(c *gin.Context) {
		if validator == nil {
			c.Next()
			return
		}

		token := bearerToken(c.GetHeader("Authorization"))
		if token == "" {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}

		matched, allowed := validator.authorize(token, c.Request.URL.Path)
		if !matched {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "unauthorized"})
			return
		}
		if !allowed {
			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{"error": "forbidden"})
			return
		}

		c.Next()
	}
}

// bearerToken extracts the token from an Authorization header value,
// stripping a leading "Bearer " prefix if present.
func bearerToken(header string) string {
	h := strings.TrimSpace(header)
	if h == "" {
		return ""
	}
	// Strip "Bearer" prefix (case-insensitive) if present.
	if len(h) >= 6 && strings.EqualFold(h[:6], "Bearer") {
		return strings.TrimSpace(h[6:])
	}
	return h
}

// apiKeysForTesting exposes the current entry count for tests.  It is not
// used in production code paths.
func (v *apiKeyValidator) apiKeysForTesting() int {
	v.mu.Lock()
	defer v.mu.Unlock()
	return len(v.entries)
}

// valid checks whether token matches one of the accepted keys, without
// considering endpoint restrictions.  Kept for backward compatibility with
// existing tests.
func (v *apiKeyValidator) valid(token string) bool {
	v.mu.Lock()
	defer v.mu.Unlock()

	if v.isFile {
		if fi, err := os.Stat(v.source); err == nil && fi.ModTime() != v.fileMod {
			if err := v.reloadFile(); err != nil {
				return slices.ContainsFunc(v.entries, func(e apiKeyEntry) bool { return e.key == token })
			}
		}
	}

	return slices.ContainsFunc(v.entries, func(e apiKeyEntry) bool { return e.key == token })
}
