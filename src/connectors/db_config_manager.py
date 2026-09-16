"""
Database Configuration Manager Module
Loads, parses, and validates database connection settings from YAML/JSON config files.
Supports environment variable interpolation, password URL encoding, and multi-engine URL synthesis.
"""
import os
import re
import json
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
import yaml


class DBConfigManager:
    """
    Manages database connection configurations.
    Enables users to provide simple connection parameters (host, port, user, etc.)
    without needing to construct complex SQLAlchemy connection strings.
    """

    @staticmethod
    def _interpolate_env_vars(text: str) -> str:
        """
        Replaces ${VAR_NAME} or ${VAR_NAME:-default} with system environment variable values.
        """
        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")

        def replace_match(match):
            var_name = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_val)

        return pattern.sub(replace_match, text)

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Loads YAML or JSON configuration file with environment variable expansion.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"데이터베이스 설정 파일을 찾을 수 없습니다: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        interpolated = cls._interpolate_env_vars(raw_content)

        if config_path.lower().endswith(".json"):
            config = json.loads(interpolated)
        else:
            config = yaml.safe_load(interpolated) or {}

        # If wrapped inside a root "database" or "db" key, unwrap it
        if "database" in config and isinstance(config["database"], dict):
            db_section = config["database"]
            # Preserve pipeline top-level overrides if present
            for k in ["table", "target", "query", "sql_file", "sample_size", "out_dir"]:
                if k in config and k not in db_section:
                    db_section[k] = config[k]
            config = db_section
        elif "db" in config and isinstance(config["db"], dict):
            db_section = config["db"]
            for k in ["table", "target", "query", "sql_file", "sample_size", "out_dir"]:
                if k in config and k not in db_section:
                    db_section[k] = config[k]
            config = db_section

        return config

    @classmethod
    def resolve_connection_url(cls, config: Dict[str, Any]) -> str:
        """
        Synthesizes a clean, standard connection URL from configuration dictionary.
        Handles direct URLs, local file paths, and structured engine parameters.
        """
        # 1. Direct URL provided
        if "url" in config and config["url"]:
            return str(config["url"]).strip()

        # 2. Direct File Path provided
        if "path" in config and config["path"]:
            path = str(config["path"]).strip()
            engine = str(config.get("engine", config.get("type", ""))).lower()
            if engine == "sqlite":
                return f"sqlite:///{path}"
            return path

        # 3. Structured RDBMS connection parameters
        engine = str(config.get("engine", config.get("type", "postgresql"))).lower().strip()
        host = config.get("host", "localhost")
        port = config.get("port")
        database = config.get("database", config.get("name", ""))
        user = config.get("username", config.get("user", ""))
        raw_password = config.get("password", "")

        # URL-encode password to safely handle special characters (@, :, /, ?, etc.)
        quoted_password = quote_plus(str(raw_password)) if raw_password else ""
        auth_part = f"{user}:{quoted_password}@" if user or quoted_password else ""

        # Engine-specific dialect and default port mapping
        if engine in ["postgres", "postgresql"]:
            port_part = f":{port}" if port else ":5432"
            schema_opt = f"?currentSchema={config['schema']}" if config.get("schema") else ""
            return f"postgresql://{auth_part}{host}{port_part}/{database}{schema_opt}"

        elif engine in ["mysql", "mariadb"]:
            port_part = f":{port}" if port else ":3306"
            return f"mysql+pymysql://{auth_part}{host}{port_part}/{database}"

        elif engine == "oracle":
            port_part = f":{port}" if port else ":1521"
            service_name = config.get("service_name", database)
            sid = config.get("sid")
            if sid:
                return f"oracle+cx_oracle://{auth_part}{host}{port_part}/?sid={sid}"
            return f"oracle+cx_oracle://{auth_part}{host}{port_part}/?service_name={service_name}"

        elif engine in ["mssql", "sqlserver"]:
            port_part = f":{port}" if port else ":1433"
            driver = config.get("driver", "ODBC Driver 17 for SQL Server")
            driver_quoted = quote_plus(driver)
            return f"mssql+pyodbc://{auth_part}{host}{port_part}/{database}?driver={driver_quoted}"

        elif engine in ["sqlite"]:
            sqlite_path = config.get("path", database or "data.db")
            return f"sqlite:///{sqlite_path}"

        elif engine in ["parquet", "excel", "csv", "json"]:
            file_path = config.get("path", database)
            return str(file_path)

        # Fallback generic format
        port_part = f":{port}" if port else ""
        return f"{engine}://{auth_part}{host}{port_part}/{database}"

    @classmethod
    def get_masked_summary(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a sanitized configuration copy with passwords masked for safe UI/logging display.
        """
        safe_copy = dict(config)
        if "password" in safe_copy and safe_copy["password"]:
            safe_copy["password"] = "********"
        if "url" in safe_copy and safe_copy["url"]:
            # Mask password inside URL if present
            safe_copy["url"] = re.sub(r":([^:@]+)@", ":********@", safe_copy["url"])
        return safe_copy
