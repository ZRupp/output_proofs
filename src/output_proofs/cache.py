"""Disk-based result caching for resumable benchmark runs."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .config import CacheConfig

logger = logging.getLogger(__name__)


def _make_cache_key(
    task_id: int,
    variant_id: str,
    transforms: list[str],
    model_name: str,
) -> str:
    """Build a deterministic SHA256 cache key.

    Transform order is sorted to ensure invariance.
    """
    payload = json.dumps(
        {
            "task_id": task_id,
            "variant_id": variant_id,
            "transforms": sorted(transforms),
            "model_name": model_name,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class ResultCache:
    """Disk-based cache for generation and translation results.

    Layout:
        {cache_dir}/generation/{hash}.json
        {cache_dir}/translation/{hash}.json
    """

    def __init__(self, config: CacheConfig, default_cache_dir: Optional[Path] = None):
        self.enabled = config.enabled
        if config.cache_dir is not None:
            self.cache_dir = config.cache_dir
        elif default_cache_dir is not None:
            self.cache_dir = default_cache_dir
        else:
            self.cache_dir = Path("results") / "cache"

        if self.enabled:
            (self.cache_dir / "generation").mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "translation").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Generation cache
    # ------------------------------------------------------------------

    def get_generation(
        self,
        task_id: int,
        variant_id: str,
        transforms: list[str],
        model_name: str,
    ) -> Optional[dict]:
        """Retrieve a cached GenerationResult dict, or None on miss/corruption."""
        if not self.enabled:
            return None
        path = self._generation_path(task_id, variant_id, transforms, model_name)
        return self._read_json(path)

    def put_generation(
        self,
        task_id: int,
        variant_id: str,
        transforms: list[str],
        model_name: str,
        result_dict: dict,
    ) -> None:
        """Write a GenerationResult dict to cache."""
        if not self.enabled:
            return
        path = self._generation_path(task_id, variant_id, transforms, model_name)
        self._write_json(path, result_dict)

    # ------------------------------------------------------------------
    # Translation cache
    # ------------------------------------------------------------------

    def get_translation(
        self,
        task_id: int,
        variant_id: str,
        transforms: list[str],
        model_name: str,
    ) -> Optional[dict]:
        """Retrieve a cached TranslationResult dict, or None on miss/corruption."""
        if not self.enabled:
            return None
        path = self._translation_path(task_id, variant_id, transforms, model_name)
        return self._read_json(path)

    def put_translation(
        self,
        task_id: int,
        variant_id: str,
        transforms: list[str],
        model_name: str,
        result_dict: dict,
    ) -> None:
        """Write a TranslationResult dict to cache."""
        if not self.enabled:
            return
        path = self._translation_path(task_id, variant_id, transforms, model_name)
        self._write_json(path, result_dict)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all cached files."""
        for subdir in ("generation", "translation"):
            cache_subdir = self.cache_dir / subdir
            if cache_subdir.exists():
                for f in cache_subdir.glob("*.json"):
                    f.unlink()
        logger.info(f"Cache cleared: {self.cache_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generation_path(
        self, task_id: int, variant_id: str, transforms: list[str], model_name: str
    ) -> Path:
        key = _make_cache_key(task_id, variant_id, transforms, model_name)
        return self.cache_dir / "generation" / f"{key}.json"

    def _translation_path(
        self, task_id: int, variant_id: str, transforms: list[str], model_name: str
    ) -> Path:
        key = _make_cache_key(task_id, variant_id, transforms, model_name)
        return self.cache_dir / "translation" / f"{key}.json"

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        try:
            if path.exists():
                return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Cache read failed for {path}: {e}")
        return None

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        try:
            path.write_text(json.dumps(data, default=str))
        except OSError as e:
            logger.warning(f"Cache write failed for {path}: {e}")
