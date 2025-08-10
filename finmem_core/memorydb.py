
import os
import pickle
import logging
from pathlib import Path
from datetime import date
from typing import List, Union, Dict, Any, Tuple, Callable, Optional
import numpy as np
from itertools import repeat
from sortedcontainers import SortedList

from .embedding import OpenAILongerThanContextEmb
from .memory_functions import (
    ImportanceScoreInitialization,
    get_importance_score_initialization_func,
    LinearImportanceScoreChange,
)
from .memory_functions import R_ConstantInitialization
from .memory_functions import LinearCompoundScore
from .memory_functions import ExponentialDecay
# ---- Repo-anchored log dir (cwd-proof) ----
BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "data" / "04_model_output_log"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Optional FAISS with pure-NumPy fallback ----
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

class _NumpyCosineIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self._vecs = None
        self._ids = None

    def add_with_ids(self, vecs: np.ndarray, ids: np.ndarray):
        if vecs is None or len(vecs) == 0:
            return
        vecs = np.array(vecs, dtype=np.float32)
        ids = np.array(ids, dtype=int)
        if self._vecs is None:
            self._vecs = vecs
            self._ids = ids
        else:
            self._vecs = np.vstack([self._vecs, vecs])
            self._ids = np.concatenate([self._ids, ids])

    def add(self, vecs: np.ndarray):
        new_ids = np.arange(len(vecs), dtype=int)
        self.add_with_ids(vecs, new_ids)

    def search(self, queries: np.ndarray, topk: int):
        if self._vecs is None or len(self._vecs) == 0:
            scores = np.zeros((len(queries), topk), dtype=np.float32)
            idxs = -np.ones((len(queries), topk), dtype=int)
            return scores, idxs
        A = self._vecs
        Q = np.array(queries, dtype=np.float32)
        # cosine sim
        def _norm(x):
            n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
            return x / n
        A_n = _norm(A)
        Q_n = _norm(Q)
        sims = Q_n @ A_n.T  # (q, n)
        best_local = np.argsort(-sims, axis=1)[:, :topk]  # local row indices
        scores = np.take_along_axis(sims, best_local, axis=1)
        # map local indices to global ids
        global_ids = self._ids if self._ids is not None else np.arange(len(A_n))
        idxs = np.take(global_ids, best_local)
        return scores, idxs

def _make_index_ip(dim: int):
    if _FAISS_AVAILABLE:
        return faiss.IndexIDMap(faiss.IndexFlatIP(dim))  # type: ignore[attr-defined]
    return _NumpyCosineIndex(dim)

# ---- ID generator ----
def id_generator_func(start: int = 0) -> Callable[[], int]:
    cur = {"v": start}
    def _next() -> int:
        cur["v"] += 1
        return cur["v"]
    return _next

class MemoryDB:
    """
    Per-layer memory with retriever and importance/recency mechanics.
    Universe structure:
      universe[symbol] = {
        "score_memory": SortedList([...], key=lambda rec: rec["important_score_recency_compound_score"]),
        "index": FAISS-or-NumPy index,
      }
    """
    def __init__(
        self,
        db_name: str,
        id_generator: Callable[[], int],
        jump_threshold_upper: float,
        jump_threshold_lower: float,
        logger: logging.Logger,
        emb_config: Dict[str, Any],
        importance_score_initialization: ImportanceScoreInitialization,
        recency_score_initialization: R_ConstantInitialization,
        compound_score_calculation: LinearCompoundScore,
        importance_score_change_access_counter: LinearImportanceScoreChange,
        decay_function: ExponentialDecay,
        clean_up_threshold_dict: Dict[str, float],
    ) -> None:
        self.db_name = db_name
        self.id_generator = id_generator
        self.jump_threshold_upper = jump_threshold_upper
        self.jump_threshold_lower = jump_threshold_lower
        self.logger = logger

        # Embedding function
        model_name = emb_config.get("model_name") or emb_config.get("model") or "text-embedding-3-small"
        chunk_char_size = int(emb_config.get("chunk_char_size", 6000))
        self.emb_config = {"model_name": model_name, "chunk_char_size": chunk_char_size}
        self.emb_func = OpenAILongerThanContextEmb(model_name=model_name, chunk_char_size=chunk_char_size)

        # Scoring functions
        self.importance_score_initialization_func = importance_score_initialization
        self.recency_score_initialization_func = recency_score_initialization
        self.compound_score_calculation_func = compound_score_calculation
        self.importance_score_change_access_counter_func = importance_score_change_access_counter
        self.decay_function = decay_function
        self.clean_up_threshold_dict = clean_up_threshold_dict

        self.universe: Dict[str, Dict[str, Any]] = {}

    def add_new_symbol(self, symbol: str) -> None:
        if symbol in self.universe:
            return
        # initialize score memory
        score_list = SortedList(key=lambda x: x["important_score_recency_compound_score"])
        # build index
        dim = self.emb_func._vector_dim() if hasattr(self.emb_func, "_vector_dim") else 1536
        index = _make_index_ip(dim)
        self.universe[symbol] = {
            "score_memory": score_list,
            "index": index,
        }

    
    def add_memory(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        """Add text memories for a symbol at a given date. Accepts various date types."""
        # Normalize date to datetime.date
        try:
            import pandas as _pd  # type: ignore
            if hasattr(date, 'to_pydatetime'):
                date = date.to_pydatetime().date()
            elif isinstance(date, _pd.Timestamp):
                date = date.to_pydatetime().date()
        except Exception:
            pass
        try:
            from datetime import datetime as _dt  # type: ignore
            if isinstance(date, _dt):
                date = date.date()
        except Exception:
            pass

        if symbol not in self.universe:
            self.add_new_symbol(symbol)
        if isinstance(text, str):
            text = [text]

        emb = self.emb_func(text)  # (n, d) np.ndarray or list->ndarray
        ids = np.array([self.id_generator() for _ in range(len(text))], dtype=int)

        # init scores
        importance_scores = [self.importance_score_initialization_func() for _ in range(len(text))]
        recency_scores = [self.recency_score_initialization_func() for _ in range(len(text))]

        # partial/compound
        partial_scores = [
            self.compound_score_calculation_func.recency_and_importance_score(
                recency_score=r, importance_score=i
            )
            for i, r in zip(importance_scores, recency_scores)
        ]

        # update index
        if _FAISS_AVAILABLE:
            # normalize for IP
            faiss.normalize_L2(emb)
            self.universe[symbol]["index"].add_with_ids(emb, ids)
        else:
            self.universe[symbol]["index"].add_with_ids(emb, ids)

        # store records
        for i_score, t, r_score, comp, the_id in zip(importance_scores, text, recency_scores, partial_scores, ids):
            self.universe[symbol]["score_memory"].add({
                "id": int(the_id),
                "text": t,
                "date": date,
                "recency_score": float(r_score),
                "importance_score": float(i_score),
                "important_score_recency_compound_score": float(comp),
                "access_count": 0,
            })

    def retrieve_top_k_text_and_ids(self, symbol: str, top_k: int, query_text: str) -> Tuple[List[str], List[int]]:
        if (symbol not in self.universe) or (len(self.universe[symbol]["score_memory"]) == 0) or (top_k == 0):
            return [], []
        top_k = min(top_k, len(self.universe[symbol]["score_memory"]))
        # Compute embedding and ensure it is 2D (n, d). Pool chunks if needed.
        import numpy as _np
        vec = self.emb_func(query_text)
        if isinstance(vec, list):
            vec = _np.array(vec, dtype=_np.float32)
        vec = _np.asarray(vec, dtype=_np.float32)
        if vec.ndim == 1:
            emb = vec[None, :]
        elif vec.ndim == 2:
            emb = vec if vec.shape[0] == 1 else vec.mean(axis=0, keepdims=True)
        else:
            emb = vec.reshape(-1, vec.shape[-1]).mean(axis=0, keepdims=True)

        if _FAISS_AVAILABLE:
            faiss.normalize_L2(emb)
        scores, idxs = self.universe[symbol]["index"].search(emb, top_k)
        ids = idxs[0].tolist()
        # map ids to texts
        id2text = {rec["id"]: rec["text"] for rec in self.universe[symbol]["score_memory"]}
        texts = [id2text.get(i, "") for i in ids if i in id2text]
        ids = [i for i in ids if i in id2text]
        return texts, ids

    def update_access_count_with_feed_back(self, symbol: str, ids: List[int], feedback_list) -> List[int]:
        """Simple access-count update; apply linear importance change, decay recency; returns successfully updated ids."""
        if symbol not in self.universe or not ids:
            return []
        # Expand scalar feedback to list
        if not isinstance(feedback_list, (list, tuple)):
            feedback_list = [float(feedback_list)] * len(ids)
        elif len(feedback_list) != len(ids):
            feedback_list = list(feedback_list) + [float(feedback_list[-1])] * (len(ids) - len(feedback_list))
            return []
        idx_by_id = {rec["id"]: rec for rec in self.universe[symbol]["score_memory"]}
        success = []
        for i, fb in zip(ids, feedback_list):
            rec = idx_by_id.get(i)
            if not rec:
                continue
            # update access count
            rec["access_count"] = rec.get("access_count", 0) + 1
            # importance shift
            try:
                rec["importance_score"] = self.importance_score_change_access_counter_func(rec["importance_score"], fb)
            except TypeError:
                # Project variant expects (access_counter, importance_score)
                rec["importance_score"] = self.importance_score_change_access_counter_func(rec.get("access_count", 1), rec["importance_score"])
            # decay recency
            new_rec, new_imp, _ = self.decay_function(rec["importance_score"], rec["access_count"])
            rec["recency_score"] = float(new_rec)
            rec["importance_score"] = float(new_imp)
            # recompute compound
            rec["important_score_recency_compound_score"] = float(
                self.compound_score_calculation_func.recency_and_importance_score(
                    recency_score=rec["recency_score"], importance_score=rec["importance_score"]
                )
            )
            success.append(i)
        # re-sort SortedList by removing and re-adding changed records
        if success:
            new_list = SortedList(key=lambda x: x["important_score_recency_compound_score"])
            for item in self.universe[symbol]["score_memory"]:
                new_list.add(item)
            self.universe[symbol]["score_memory"] = new_list
        return success

    # ---- Persistence ----
    def save_checkpoint(self, path: str, force: bool = False) -> None:
        path = os.path.join(path, self.db_name)
        os.makedirs(path, exist_ok=True)
        # save state dict
        state_dict = {
            "db_name": self.db_name,
            "emb_config": self.emb_config,
            "jump_threshold_upper": self.jump_threshold_upper,
            "jump_threshold_lower": self.jump_threshold_lower,
            "clean_up_threshold_dict": self.clean_up_threshold_dict,
        }
        with open(os.path.join(path, "state_dict.pkl"), "wb") as f:
            pickle.dump(state_dict, f)
        # save universe
        save_universe = {}
        for sym, record in self.universe.items():
            if _FAISS_AVAILABLE:
                idx_path = os.path.join(path, f"{sym}.index")
                faiss.write_index(record["index"], idx_path)  # type: ignore[arg-type]
            else:
                idx_path = os.path.join(path, f"{sym}.npz")
                vecs = record["index"]._vecs if hasattr(record["index"], "_vecs") else None
                ids = record["index"]._ids if hasattr(record["index"], "_ids") else None
                np.savez(idx_path, vecs=vecs, ids=ids)
            save_universe[sym] = {
                "score_memory": list(record["score_memory"]),
                "index_save_path": idx_path,
                "faiss": _FAISS_AVAILABLE,
            }
        with open(os.path.join(path, "universe_index.pkl"), "wb") as f:
            pickle.dump(save_universe, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "MemoryDB":
        """
        Load a MemoryDB from `path` supporting:
          - flat:   <path>/state_dict.pkl + <path>/universe_index.pkl
          - nested: <path>/<subdir>/state_dict.pkl + <path>/<subdir>/universe_index.pkl
        The memory layer is inferred from the last folder name in `path`.
        """
        import os
        from pathlib import Path
        from finmem_core.memory_functions import (
            get_importance_score_initialization_func,
            R_ConstantInitialization,
            LinearCompoundScore,
            LinearImportanceScoreChange,
            ExponentialDecay,
        )

        base = Path(path)
        layer_folder = base.name  # short_term_memory | mid_term_memory | long_term_memory | reflection_memory

        # Prefer direct files under `path`
        sd = base / "state_dict.pkl"
        ui = base / "universe_index.pkl"

        # If not found, search exactly one level down for a subdir with the files
        if not sd.exists() or not ui.exists():
            candidate = None
            if base.exists():
                for d in base.iterdir():
                    if d.is_dir() and (d / "state_dict.pkl").exists() and (d / "universe_index.pkl").exists():
                        candidate = d
                        break
            if candidate is None:
                raise FileNotFoundError(f"state_dict.pkl not found under {base} (flat or nested).")
            sd = candidate / "state_dict.pkl"
            ui = candidate / "universe_index.pkl"

        # Load state + universe
        with open(sd, "rb") as f:
            state_dict = pickle.load(f)
        with open(ui, "rb") as f:
            universe = pickle.load(f)

        # Infer layer key for importance init
        layer_map = {
            "short_term_memory": "short",
            "mid_term_memory": "mid",
            "long_term_memory": "long",
            "reflection_memory": "reflection",
        }
        layer_key = layer_map.get(layer_folder, "short")

        obj = cls(
            db_name=state_dict.get("db_name", "memory_db"),
            id_generator=id_generator_func(),
            jump_threshold_upper=state_dict.get("jump_threshold_upper", 0.0),
            jump_threshold_lower=state_dict.get("jump_threshold_lower", 0.0),
            logger=logging.getLogger(__name__),
            emb_config=state_dict.get("emb_config", {"model_name": "text-embedding-3-small", "chunk_char_size": 6000}),
            importance_score_initialization=get_importance_score_initialization_func(type="sample", memory_layer=layer_key),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(),
            clean_up_threshold_dict=state_dict.get("clean_up_threshold_dict", {}),
        )

        # Restore stored universe
        obj.universe = universe or {}
        return obj

class BrainDB:
    def __init__(
        self,
        agent_name: str,
        emb_config: Dict[str, Any],
        id_generator: Callable[[], int],
        short_term_memory: MemoryDB,
        mid_term_memory: MemoryDB,
        long_term_memory: MemoryDB,
        reflection_memory: MemoryDB,
        logger: logging.Logger,
        use_gpu: bool = True,
    ):
        self.agent_name = agent_name
        self.emb_config = emb_config
        self.use_gpu = use_gpu
        self.id_generator = id_generator
        self.short_term_memory = short_term_memory
        self.mid_term_memory = mid_term_memory
        self.long_term_memory = long_term_memory
        self.reflection_memory = reflection_memory
        self.logger = logger
        self.read_only = False
        self._ephemeral_short: List[Tuple[str, str]] = []
        self._ephemeral_date = None

    # --- UI-friendly defaults for Streamlit cfg ---
    @staticmethod
    def _layer_defaults(name: str) -> Dict[str, Any]:
        base = {
            "jump_threshold_upper": 999999999.0,
            "jump_threshold_lower": -999999999.0,
            "importance_score_initialization": "sample",
            "decay_params": {},
            "clean_up_threshold_dict": {"recency_threshold": 0.05, "importance_threshold": 50.0},
        }
        return base

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BrainDB":
        # other states
        id_gen = id_generator_func()
        agent_name = config["general"]["agent_name"]
        symbol = config["general"]["trading_symbol"]

        # logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logging_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        log_path = LOG_DIR / f"{symbol}_run.log"
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setFormatter(logging_formatter)
        if not any(getattr(h, 'baseFilename', None) == str(log_path) for h in logger.handlers):
            logger.addHandler(file_handler)

        # embedding config
        emb_config = (
            config.get("embedding", {})
            or {"model_name": "text-embedding-3-small", "chunk_char_size": 6000}
        )

        # per-layer merged cfg
        def merged(name: str) -> Dict[str, Any]:
            d = dict(BrainDB._layer_defaults(name))
            d.update(config.get(name, {}))
            return d
        short_cfg = merged("short")
        mid_cfg = merged("mid")
        long_cfg = merged("long")
        reflection_cfg = merged("reflection")

        # memory layers
        short_term_memory = MemoryDB(
            db_name=f"{agent_name}_short",
            id_generator=id_gen,
            emb_config=emb_config,
            jump_threshold_upper=short_cfg.get("jump_threshold_upper"),
            jump_threshold_lower=short_cfg.get("jump_threshold_lower"),
            importance_score_initialization=get_importance_score_initialization_func(
                type=short_cfg.get("importance_score_initialization", "sample"), memory_layer="short"
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**short_cfg.get("decay_params", {})),
            clean_up_threshold_dict=short_cfg.get("clean_up_threshold_dict", {}),
            logger=logger,
        )
        mid_term_memory = MemoryDB(
            db_name=f"{agent_name}_mid",
            id_generator=id_gen,
            emb_config=emb_config,
            jump_threshold_upper=mid_cfg.get("jump_threshold_upper"),
            jump_threshold_lower=mid_cfg.get("jump_threshold_lower"),
            importance_score_initialization=get_importance_score_initialization_func(
                type=mid_cfg.get("importance_score_initialization", "sample"), memory_layer="mid"
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**mid_cfg.get("decay_params", {})),
            clean_up_threshold_dict=mid_cfg.get("clean_up_threshold_dict", {}),
            logger=logger,
        )
        long_term_memory = MemoryDB(
            db_name=f"{agent_name}_long",
            id_generator=id_gen,
            emb_config=emb_config,
            jump_threshold_upper=long_cfg.get("jump_threshold_upper"),
            jump_threshold_lower=long_cfg.get("jump_threshold_lower"),
            importance_score_initialization=get_importance_score_initialization_func(
                type=long_cfg.get("importance_score_initialization", "sample"), memory_layer="long"
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**long_cfg.get("decay_params", {})),
            clean_up_threshold_dict=long_cfg.get("clean_up_threshold_dict", {}),
            logger=logger,
        )
        reflection_memory = MemoryDB(
            db_name=f"{agent_name}_reflection",
            id_generator=id_gen,
            emb_config=emb_config,
            jump_threshold_upper=reflection_cfg.get("jump_threshold_upper"),
            jump_threshold_lower=reflection_cfg.get("jump_threshold_lower"),
            importance_score_initialization=get_importance_score_initialization_func(
                type=reflection_cfg.get("importance_score_initialization", "sample"), memory_layer="reflection"
            ),
            recency_score_initialization=R_ConstantInitialization(),
            compound_score_calculation=LinearCompoundScore(),
            importance_score_change_access_counter=LinearImportanceScoreChange(),
            decay_function=ExponentialDecay(**reflection_cfg.get("decay_params", {})),
            clean_up_threshold_dict=reflection_cfg.get("clean_up_threshold_dict", {}),
            logger=logger,
        )

        return cls(
            emb_config=emb_config,
            agent_name=agent_name,
            id_generator=id_gen,
            short_term_memory=short_term_memory,
            mid_term_memory=mid_term_memory,
            long_term_memory=long_term_memory,
            reflection_memory=reflection_memory,
            logger=logger,
        )

    # --- convenience passthroughs used by the agent ---
    def add_memory_short(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.short_term_memory.add_memory(symbol, date, text)

    def add_memory_mid(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.mid_term_memory.add_memory(symbol, date, text)

    def add_memory_long(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.long_term_memory.add_memory(symbol, date, text)

    def add_memory_reflection(self, symbol: str, date: date, text: Union[List[str], str]) -> None:
        self.reflection_memory.add_memory(symbol, date, text)

    def retrieve_memory_short(self, symbol: str, top_k: int, query_text: str) -> Tuple[List[str], List[int]]:
        return self.short_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)

    def retrieve_memory_mid(self, symbol: str, top_k: int, query_text: str) -> Tuple[List[str], List[int]]:
        return self.mid_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)

    def retrieve_memory_long(self, symbol: str, top_k: int, query_text: str) -> Tuple[List[str], List[int]]:
        return self.long_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)

    def retrieve_memory_reflection(self, symbol: str, top_k: int, query_text: str) -> Tuple[List[str], List[int]]:
        return self.reflection_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_short(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.short_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_mid(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.mid_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_long(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.long_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_reflection(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.reflection_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_short(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.short_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_mid(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.mid_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_long(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.long_term_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)
    def query_reflection(self, query_text: str = "", top_k: int = 5, symbol: str = ""):
        """Agent-facing alias. Returns (texts, ids)."""
        return self.reflection_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)

        return self.reflection_memory.retrieve_top_k_text_and_ids(symbol, top_k, query_text)

    def update_access_count_with_feed_back(self, symbol: str, ids: List[int], feedback: float) -> None:
        # propagate update across layers (short -> mid -> long) similar to original behavior
        ids_list = list(ids)
        if ids_list:
            success = self.short_term_memory.update_access_count_with_feed_back(symbol, ids_list, list(repeat(feedback, len(ids_list))))
            ids_list = [i for i in ids_list if i not in success]
        if ids_list:
            success = self.mid_term_memory.update_access_count_with_feed_back(symbol, ids_list, list(repeat(feedback, len(ids_list))))
            ids_list = [i for i in ids_list if i not in success]
        if ids_list:
            self.long_term_memory.update_access_count_with_feed_back(symbol, ids_list, list(repeat(feedback, len(ids_list))))

    # persistence for the whole brain
    
    def _decay_clean_promote(self, layer: "MemoryDB", next_layer: Optional["MemoryDB"]) -> None:
        """
        Apply one time-step of decay and cleanup on a memory layer.
        Optionally promote high-scoring items to the next layer.
        """
        # thresholds
        rec_thr = float(layer.clean_up_threshold_dict.get("recency_threshold", 0.05))
        imp_thr = float(layer.clean_up_threshold_dict.get("importance_threshold", 50.0))
        jump_up = float(layer.jump_threshold_upper)

        for symbol in list(layer.universe.keys()):
            slist = layer.universe[symbol]["score_memory"]
            items = list(slist)  # snapshot
            kept = []
            promotions = []

            for rec in items:
                # decay one tick
                new_rec, new_imp, new_delta = layer.decay_function(rec["importance_score"], rec.get("access_count", 0))
                rec["recency_score"] = float(new_rec)
                rec["importance_score"] = float(new_imp)
                rec["access_count"] = float(new_delta)

                # recompute compound
                rec["important_score_recency_compound_score"] = float(
                    layer.compound_score_calculation_func.recency_and_importance_score(
                        recency_score=rec["recency_score"], importance_score=rec["importance_score"]
                    )
                )

                # check promotion vs cleanup
                if next_layer is not None and rec["important_score_recency_compound_score"] > jump_up:
                    promotions.append(rec)
                    continue

                # cleanup only if both signals are low
                if rec["recency_score"] < rec_thr and rec["importance_score"] < imp_thr:
                    # drop
                    continue

                kept.append(rec)

            # rebuild sorted list
            new_list = SortedList(key=lambda x: x["important_score_recency_compound_score"])
            for r in kept:
                new_list.add(r)
            layer.universe[symbol]["score_memory"] = new_list

            # handle promotions
            if next_layer is not None and promotions:
                for r in promotions:
                    try:
                        next_layer.add_memory(symbol=symbol, date=r.get("date"), text=r.get("text", ""))
                    except Exception:
                        # be resilient: if date causes trouble, pass today
                        from datetime import date as _date
                        next_layer.add_memory(symbol=symbol, date=_date.today(), text=r.get("text", ""))

    
    def step(self) -> None:
        """
        Advance memory by one training tick:
        - Decay & cleanup in each layer
        - Promote high-scoring items short->mid->long
        (Reflection layer is maintained but not promoted.)
        """
        # process in order: short -> mid -> long; reflection has no promotion target
        self._decay_clean_promote(self.short_term_memory, self.mid_term_memory)
        self._decay_clean_promote(self.mid_term_memory, self.long_term_memory)
        self._decay_clean_promote(self.long_term_memory, None)
        # reflection: decay + cleanup only
        self._decay_clean_promote(self.reflection_memory, None)
    
    def save_checkpoint(self, path: str, force: bool = False) -> None:
        os.makedirs(path, exist_ok=True)
        # save memory layers
        self.short_term_memory.save_checkpoint(os.path.join(path, "short_term_memory"), force=force)
        self.mid_term_memory.save_checkpoint(os.path.join(path, "mid_term_memory"), force=force)
        self.long_term_memory.save_checkpoint(os.path.join(path, "long_term_memory"), force=force)
        self.reflection_memory.save_checkpoint(os.path.join(path, "reflection_memory"), force=force)
        # save brain-level state
        state = {
            "agent_name": self.agent_name,
            "emb_config": self.emb_config,
        }
        with open(os.path.join(path, "brain_state.pkl"), "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_checkpoint(cls, path: str) -> "BrainDB":
        with open(os.path.join(path, "brain_state.pkl"), "rb") as f:
            state = pickle.load(f)
        id_gen = id_generator_func()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_path = LOG_DIR / f"brain_{state.get('agent_name','agent')}.log"
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(fh)

        stm = MemoryDB.load_checkpoint(os.path.join(path, "short_term_memory"))
        mtm = MemoryDB.load_checkpoint(os.path.join(path, "mid_term_memory"))
        ltm = MemoryDB.load_checkpoint(os.path.join(path, "long_term_memory"))
        rfm = MemoryDB.load_checkpoint(os.path.join(path, "reflection_memory"))
        return cls(
            agent_name=state.get("agent_name", "agent_1"),
            emb_config=state.get("emb_config", {"model_name":"text-embedding-3-small","chunk_char_size":6000}),
            id_generator=id_gen,
            short_term_memory=stm,
            mid_term_memory=mtm,
            long_term_memory=ltm,
            reflection_memory=rfm,
            logger=logger,
        )
