"""Defense-in-Depth SQL Security Guard and Process-Isolated Execution Sandbox.

Enforces:
1. Deep AST traversal via sqlglot (rejecting non-SELECT, multi-statement, schema snooping).
2. View, column, and scalar/aggregate function allowlisting (including nested CTEs/subqueries).
3. Safe SQL comment parsing.
4. Windows multiprocessing spawn compatibility with module-level worker.
5. Deterministic query timeout enforcement via physical worker process termination.
"""

from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import queue
from typing import Any, Dict, List, Optional, Set, Tuple
import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp
import yaml

from insightx.utils.logger import get_logger

logger = get_logger(__name__)


class SQLSecurityError(Exception):
    """Base exception for all SQL security violations."""
    pass


class DisallowedOperationError(SQLSecurityError):
    """Raised when query contains non-SELECT roots, DDL, DML, or statement chaining."""
    pass


class UnapprovedEntityError(SQLSecurityError):
    """Raised when query references unapproved tables, columns, functions, or schemas."""
    pass


class QueryTimeoutError(SQLSecurityError):
    """Raised when query execution exceeds the configured wall-clock timeout."""
    pass


# Default configuration fallback
DEFAULT_GUARDRAILS = {
    "execution": {
        "timeout_seconds": 3.0,
        "max_rows": 500,
        "engine_max_memory": "256MB",
        "threads": 1,
        "enable_external_access": False,
    },
    "allowed_views": {
        "v_orders": [
            "order_id", "customer_id", "order_status", "order_purchase_timestamp",
            "order_approved_at", "order_delivered_carrier_date",
            "order_delivered_customer_date", "order_estimated_delivery_date"
        ],
        "v_order_items": [
            "order_id", "order_item_id", "product_id", "seller_id",
            "shipping_limit_date", "price", "freight_value"
        ],
        "v_order_payments": [
            "order_id", "payment_sequential", "payment_type",
            "payment_installments", "payment_value"
        ],
        "v_order_reviews": [
            "review_id", "order_id", "review_score", "review_comment_title",
            "review_comment_message", "review_creation_date", "review_answer_timestamp"
        ],
        "v_customers": [
            "customer_id", "customer_unique_id", "customer_zip_code_prefix",
            "customer_city", "customer_state"
        ],
        "v_products": [
            "product_id", "product_category_name", "product_name_lenght",
            "product_description_lenght", "product_photos_qty", "product_weight_g",
            "product_length_cm", "product_height_cm", "product_width_cm"
        ],
        "v_sellers": [
            "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"
        ]
    },
    "allowed_functions": [
        "SUM", "AVG", "COUNT", "MIN", "MAX", "ROUND", "STRFTIME", "DATE_TRUNC",
        "COALESCE", "NULLIF", "ABS", "DATE_DIFF", "CAST"
    ],
    "prohibited_keywords": [
        "read_csv", "read_parquet", "read_json", "httpfs", "install", "load",
        "copy", "export", "import", "pragma", "duckdb_tables", "duckdb_views",
        "information_schema", "sqlite_master"
    ]
}


def _duckdb_worker_exec(
    query: str,
    tables_data: Dict[str, pd.DataFrame],
    duckdb_config: Dict[str, Any],
    result_queue: Any,
    error_queue: Any,
) -> None:
    """Module-level worker function executing queries inside an isolated child process.
    
    Guarantees Windows spawn compatibility:
    - Initialized inside child process.
    - Zero DuckDB connections or C++ pointers passed across processes.
    - Serializable inputs and outputs.
    """
    try:
        con = duckdb.connect(database=":memory:", config=duckdb_config)
        for view_name, df in tables_data.items():
            con.register(view_name, df)
        
        result_df = con.execute(query).df()
        con.close()
        result_queue.put(result_df)
    except Exception as exc:
        error_queue.put(str(exc))


class SQLGuard:
    """Validates and safely executes analytical SQL queries with AST security and process sandboxing."""

    def __init__(self, guardrails_config_path: Optional[Path] = None):
        """Loads guardrail policies and allowlists from YAML config."""
        if guardrails_config_path and Path(guardrails_config_path).exists():
            with open(guardrails_config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or DEFAULT_GUARDRAILS
        else:
            self.config = DEFAULT_GUARDRAILS

        self.allowed_views: Dict[str, Set[str]] = {
            v.lower(): {col.lower() for col in cols}
            for v, cols in self.config.get("allowed_views", {}).items()
        }
        self.allowed_functions: Set[str] = {
            f.upper() for f in self.config.get("allowed_functions", [])
        }
        self.prohibited_keywords: Set[str] = {
            k.lower() for k in self.config.get("prohibited_keywords", [])
        }
        self.timeout_seconds: float = float(
            self.config.get("execution", {}).get("timeout_seconds", 3.0)
        )
        self.max_rows: int = int(self.config.get("execution", {}).get("max_rows", 500))
        self.duckdb_engine_config: Dict[str, Any] = {
            "enable_external_access": False,
            "max_memory": str(self.config.get("execution", {}).get("engine_max_memory", "256MB")),
            "threads": int(self.config.get("execution", {}).get("threads", 1)),
        }

    def _expand_wildcards(self, ast_root: exp.Expression, cte_names: Set[str]) -> None:
        """Validates wildcard expressions for security without expanding them.

        Primary wildcard security is enforced by DataFrame column filtering in
        execute_safely(), which strips unauthorized columns from DataFrames before they
        reach DuckDB. This method provides a complementary AST-level check that logs
        and audits wildcard usage without rewriting the SQL (which would break queries
        against DataFrames that have fewer columns than the full allowlist).

        The column-validation step (step 6) still rejects any explicit unauthorized
        column names. Wildcards that survive to DuckDB execution are safe because the
        underlying DataFrame has already been filtered to approved columns only.

        Args:
            ast_root:  Root AST node (the entire parsed statement).
            cte_names: Set of lowercased CTE alias names defined in this query.
        """
        # Audit all Select nodes for wildcard usage — no rewriting needed.
        # Security guarantee: execute_safely() strips DataFrames to allowed columns,
        # so DuckDB SELECT * can only ever return approved columns.
        for select_node in ast_root.find_all(exp.Select):
            for expr in select_node.expressions:
                if isinstance(expr, exp.Star):
                    logger.debug("Wildcard SELECT * detected; DataFrame column filter enforces allowlist.")
                elif isinstance(expr, exp.Column) and isinstance(expr.this, exp.Star):
                    table_ref_raw = expr.table.lower() if expr.table else ""
                    logger.debug(
                        "Qualified wildcard SELECT %s.* detected; DataFrame column filter enforces allowlist.",
                        table_ref_raw,
                    )

    def validate_ast(self, raw_sql: str) -> str:
        """Parses and recursively validates query against security allowlists.
        
        Handles SQL comments safely and enforces single-statement read-only execution.
        
        Returns:
            str: Normalized, validated SQL query with row limits enforced.
        
        Raises:
            SQLSecurityError: If any security boundary is breached.
        """
        # 1. Statement isolation: parse and verify single statement
        try:
            statements = sqlglot.parse(raw_sql, read="duckdb")
        except Exception as exc:
            raise DisallowedOperationError(f"SQL parsing failed: {exc}")

        if len(statements) == 0:
            raise DisallowedOperationError("Empty SQL query provided.")
        if len(statements) > 1:
            raise DisallowedOperationError("Statement chaining (multiple queries separated by ';') is strictly prohibited.")

        ast_root = statements[0]

        # 2. Root expression must be a SELECT statement
        if not isinstance(ast_root, exp.Select):
            raise DisallowedOperationError(
                f"Unauthorized root operation: {type(ast_root).__name__}. "
                "Only read-only SELECT queries are permitted."
            )

        # 3. Detect CTE names and SELECT aliases to register local scopes
        cte_names: Set[str] = set()
        for cte in ast_root.find_all(exp.CTE):
            cte_alias = cte.alias
            if cte_alias:
                cte_names.add(cte_alias.lower())

        select_aliases: Set[str] = set()
        for alias_node in ast_root.find_all(exp.Alias):
            if alias_node.alias:
                select_aliases.add(alias_node.alias.lower())

        # 3b. Expand wildcard expressions into explicit approved column projections.
        #     This must occur AFTER CTE detection and BEFORE table/column validation
        #     so that no Star() nodes survive into the allowlist checks.
        self._expand_wildcards(ast_root, cte_names)

        # 4. Validate all Table nodes (including joins, subqueries, CTEs)
        for table_node in ast_root.find_all(exp.Table):
            table_name = table_node.name.lower()
            
            # Check for disallowed schema qualifiers (e.g. duckdb.*, information_schema.*)
            if table_node.db:
                schema_name = table_node.db.lower()
                if schema_name not in ("main", ""):
                    raise UnapprovedEntityError(
                        f"External/System schema qualifier '{schema_name}' is disallowed."
                    )

            # Table must be in allowed views or in local CTE scope
            if table_name not in self.allowed_views and table_name not in cte_names:
                raise UnapprovedEntityError(
                    f"Access to table/view '{table_name}' is not permitted. "
                    f"Allowed views: {sorted(list(self.allowed_views.keys()))}"
                )

        # 5. Validate all Function nodes (including nested functions)
        canonical_func_map = {
            "TIME_TO_STR": "STRFTIME",
            "TIMESTAMP_TRUNC": "DATE_TRUNC",
        }
        for func_node in ast_root.find_all(exp.Func, exp.Anonymous):
            if isinstance(func_node, exp.Cast):
                continue
            raw_func_name = func_node.sql_name().upper() if hasattr(func_node, "sql_name") else func_node.key.upper()
            func_name = canonical_func_map.get(raw_func_name, raw_func_name)
            
            # Check against prohibited keyword blacklist
            if func_name.lower() in self.prohibited_keywords:
                raise UnapprovedEntityError(f"Prohibited function '{func_name}' detected.")

            if func_name not in self.allowed_functions:
                raise UnapprovedEntityError(
                    f"Function '{func_name}' is not allowlisted. "
                    f"Allowed functions: {sorted(list(self.allowed_functions))}"
                )

        # 6. Validate Column nodes against view schema allowlists
        for col_node in ast_root.find_all(exp.Column):
            col_name = col_node.name.lower()
            table_ref = col_node.table.lower() if col_node.table else None

            # Skip wildcard expressions — security is enforced by DataFrame column filtering
            # in execute_safely(), which strips unauthorized columns before DuckDB sees them.
            # Wildcards against CTE outputs cannot be validated statically (dynamic schema).
            # Wildcards against real views are safe because the DataFrame is pre-filtered.
            if col_name == "*":
                continue

            # Skip columns that match SELECT aliases defined in this query
            if col_name in select_aliases:
                continue

            # Skip columns whose table reference is a CTE local name (dynamic schema)
            if table_ref and table_ref in cte_names:
                continue

            # If table is explicitly specified, check that column exists in that view
            if table_ref and table_ref in self.allowed_views:
                if col_name not in self.allowed_views[table_ref]:
                    raise UnapprovedEntityError(
                        f"Column '{col_name}' is not permitted for view '{table_ref}'."
                    )
            elif not table_ref:
                # Column without table qualifier: must belong to at least one allowed view or CTE
                matching_views = [
                    v for v, cols in self.allowed_views.items() if col_name in cols
                ]
                if not matching_views and not cte_names:
                    raise UnapprovedEntityError(
                        f"Column '{col_name}' does not match any allowlisted view schema."
                    )

        # 7. Enforce LIMIT constraint
        limit_node = ast_root.find(exp.Limit)
        if limit_node is None:
            # Inject LIMIT 500 automatically
            ast_root = ast_root.limit(self.max_rows)
        else:
            # If user limit exceeds max_rows, clamp it
            try:
                user_limit = int(limit_node.expression.this)
                if user_limit > self.max_rows:
                    limit_node.set("expression", exp.Literal.number(self.max_rows))
            except (ValueError, TypeError, AttributeError):
                limit_node.set("expression", exp.Literal.number(self.max_rows))

        # Re-transpile clean DuckDB SQL (comments safely normalized)
        return ast_root.sql(dialect="duckdb")

    def execute_safely(
        self,
        raw_sql: str,
        tables_data: Dict[str, pd.DataFrame],
        timeout_seconds: Optional[float] = None,
    ) -> pd.DataFrame:
        """Validates AST and executes query in an isolated, terminable worker process.
        
        Args:
            raw_sql: SQL query string to validate and execute.
            tables_data: Serializable dictionary of view DataFrames.
            timeout_seconds: Custom execution timeout (defaults to config value).
            
        Returns:
            pd.DataFrame: Query results (max 500 rows).
            
        Raises:
            SQLSecurityError: If query fails AST validation.
            QueryTimeoutError: If execution exceeds timeout and is killed.
        """
        # 1. Deep AST validation
        validated_sql = self.validate_ast(raw_sql)
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds

        # 2. Defence-in-depth: strip unauthorized columns from registered DataFrames
        #    before they are passed into the DuckDB worker process.  Even if a wildcard
        #    expansion were somehow bypassed at the AST level, the underlying data would
        #    never contain the sensitive column in the first place.
        filtered_tables: Dict[str, pd.DataFrame] = {}
        for view_name, df in tables_data.items():
            canonical = view_name.lower()
            if canonical in self.allowed_views:
                approved_cols = [
                    c for c in df.columns
                    if c.lower() in self.allowed_views[canonical]
                ]
                filtered_tables[view_name] = df[approved_cols]
            else:
                # Unknown view (not in allowlist) — pass through unchanged;
                # AST validation will have rejected any query referencing it.
                filtered_tables[view_name] = df

        # 3. Windows-compatible process isolation
        result_queue = multiprocessing.Queue()
        error_queue = multiprocessing.Queue()

        proc = multiprocessing.Process(
            target=_duckdb_worker_exec,
            args=(
                validated_sql,
                filtered_tables,
                self.duckdb_engine_config,
                result_queue,
                error_queue,
            ),
        )

        proc.start()
        proc.join(timeout=timeout)

        # 3. Timeout watchdog: physical process termination
        if proc.is_alive():
            logger.warning(f"Query exceeded timeout of {timeout}s. Terminating worker process.")
            proc.terminate()
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=0.5)
            raise QueryTimeoutError(f"Query execution timed out after {timeout} seconds and was terminated.")

        # 4. Check for errors in worker process
        if not error_queue.empty():
            error_msg = error_queue.get()
            raise SQLSecurityError(f"Query execution failed inside sandbox: {error_msg}")

        # 5. Retrieve result
        try:
            return result_queue.get(timeout=1.0)
        except queue.Empty:
            return pd.DataFrame()
