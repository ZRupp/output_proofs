"""Tests for disk-based result caching."""

import pytest

from output_proofs.cache import ResultCache, _make_cache_key
from output_proofs.config import CacheConfig


class TestCacheKey:
    """Tests for cache key generation."""

    def test_deterministic(self):
        """Same inputs produce the same key."""
        key1 = _make_cache_key(101, "101_v0", ["v_noise_rename"], "gpt2")
        key2 = _make_cache_key(101, "101_v0", ["v_noise_rename"], "gpt2")
        assert key1 == key2

    def test_different_inputs_different_keys(self):
        """Different inputs produce different keys."""
        key1 = _make_cache_key(101, "101_v0", ["v_noise_rename"], "gpt2")
        key2 = _make_cache_key(102, "102_v0", ["v_noise_rename"], "gpt2")
        assert key1 != key2

    def test_different_model_different_key(self):
        """Different model names produce different keys."""
        key1 = _make_cache_key(101, "101_v0", ["v_noise_rename"], "gpt2")
        key2 = _make_cache_key(101, "101_v0", ["v_noise_rename"], "llama-3b")
        assert key1 != key2

    def test_transform_order_invariance(self):
        """Transform order does not affect the key."""
        key1 = _make_cache_key(101, "101_v0", ["v_noise_rename", "param_reorder"], "gpt2")
        key2 = _make_cache_key(101, "101_v0", ["param_reorder", "v_noise_rename"], "gpt2")
        assert key1 == key2

    def test_key_is_hex_string(self):
        """Cache key is a valid hex SHA256 hash."""
        key = _make_cache_key(101, "101_v0", ["v_noise_rename"], "gpt2")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


class TestResultCache:
    """Tests for ResultCache read/write operations."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Provide a temporary cache directory."""
        return tmp_path / "test_cache"

    @pytest.fixture
    def cache(self, cache_dir):
        """Create an enabled cache."""
        config = CacheConfig(enabled=True, cache_dir=cache_dir)
        return ResultCache(config)

    @pytest.fixture
    def disabled_cache(self, cache_dir):
        """Create a disabled cache."""
        config = CacheConfig(enabled=False, cache_dir=cache_dir)
        return ResultCache(config)

    @pytest.fixture
    def sample_gen_result(self):
        return {
            "prompt": "Write a function",
            "generated_code": "def add(a, b): return a + b",
            "full_response": "def add(a, b): return a + b",
            "success": True,
            "error": None,
            "tokens_generated": 10,
        }

    @pytest.fixture
    def sample_trans_result(self):
        return {
            "python_code": "def add(a, b): return a + b",
            "lean_code": "def add (a b : Nat) : Nat := a + b",
            "success": True,
            "error": None,
            "prompt_used": "Translate...",
        }

    def test_generation_put_get_roundtrip(self, cache, sample_gen_result):
        """Put then get returns the same data."""
        cache.put_generation(101, "101_v0", ["v_noise_rename"], "gpt2", sample_gen_result)
        result = cache.get_generation(101, "101_v0", ["v_noise_rename"], "gpt2")
        assert result == sample_gen_result

    def test_translation_put_get_roundtrip(self, cache, sample_trans_result):
        """Put then get returns the same data for translation."""
        cache.put_translation(101, "101_v0", ["v_noise_rename"], "model", sample_trans_result)
        result = cache.get_translation(101, "101_v0", ["v_noise_rename"], "model")
        assert result == sample_trans_result

    def test_cache_miss_returns_none(self, cache):
        """Get on missing key returns None."""
        result = cache.get_generation(999, "999_v0", [], "gpt2")
        assert result is None

    def test_disabled_cache_returns_none(self, disabled_cache, sample_gen_result):
        """Disabled cache always returns None."""
        disabled_cache.put_generation(101, "101_v0", [], "gpt2", sample_gen_result)
        result = disabled_cache.get_generation(101, "101_v0", [], "gpt2")
        assert result is None

    def test_clear_removes_files(self, cache, sample_gen_result, sample_trans_result):
        """Clear removes all cached files."""
        cache.put_generation(101, "101_v0", [], "gpt2", sample_gen_result)
        cache.put_translation(101, "101_v0", [], "model", sample_trans_result)

        cache.clear()

        assert cache.get_generation(101, "101_v0", [], "gpt2") is None
        assert cache.get_translation(101, "101_v0", [], "model") is None

    def test_corrupted_file_returns_none(self, cache, cache_dir):
        """Corrupted JSON file returns None instead of raising."""
        # Write a corrupted file
        key = _make_cache_key(101, "101_v0", ["rename"], "gpt2")
        corrupt_path = cache_dir / "generation" / f"{key}.json"
        corrupt_path.write_text("{invalid json")

        result = cache.get_generation(101, "101_v0", ["rename"], "gpt2")
        assert result is None
