"""
Unit & Integration Tests for DBConfigManager.
Verifies:
1. Structured YAML and JSON database configuration loading.
2. Environment variable interpolation (${VAR:-default}).
3. Special character URL-encoding in database credentials.
4. Multi-engine connection URL synthesis (Postgres, MySQL, Oracle, MSSQL, SQLite, Parquet).
5. Safe credential masking for logging and UI display.
6. E2E pipeline execution with --config parameter.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import yaml
import json
from src.connectors.db_config_manager import DBConfigManager
from src.sdk import AutoDataAnalyzer


def test_env_var_interpolation(tmp_path):
    """Verifies that ${ENV_VAR} and default fallbacks ${VAR:-fallback} are resolved properly."""
    os.environ["TEST_SECURE_PASS"] = "P@ssw0rd!123"

    config_yaml = """
    database:
      engine: postgresql
      host: 10.0.0.1
      port: 5432
      database: app_db
      username: admin
      password: ${TEST_SECURE_PASS}
      schema: ${NON_EXISTENT_SCHEMA:-public}
    """
    cfg_file = os.path.join(tmp_path, "test_env.yaml")
    with open(cfg_file, "w", encoding="utf-8") as f:
        f.write(config_yaml)

    config = DBConfigManager.load_config(cfg_file)
    assert config["password"] == "P@ssw0rd!123"
    assert config["schema"] == "public"

    url = DBConfigManager.resolve_connection_url(config)
    assert "P%40ssw0rd%21123" in url  # @ and ! properly URL encoded
    assert "currentSchema=public" in url


def test_multi_engine_url_synthesis():
    """Verifies connection URL synthesis across PostgreSQL, MySQL, Oracle, MS-SQL, and SQLite."""
    # 1. PostgreSQL
    pg_cfg = {"engine": "postgresql", "host": "db.corp", "port": 5432, "database": "dw", "username": "user", "password": "pwd"}
    assert DBConfigManager.resolve_connection_url(pg_cfg) == "postgresql://user:pwd@db.corp:5432/dw"

    # 2. MySQL
    my_cfg = {"engine": "mysql", "host": "localhost", "port": 3306, "database": "shop", "username": "root", "password": "123"}
    assert DBConfigManager.resolve_connection_url(my_cfg) == "mysql+pymysql://root:123@localhost:3306/shop"

    # 3. Oracle (Service Name)
    ora_cfg = {"engine": "oracle", "host": "ora.local", "port": 1521, "database": "XEPDB1", "username": "scott", "password": "tiger"}
    assert "service_name=XEPDB1" in DBConfigManager.resolve_connection_url(ora_cfg)

    # 4. MS-SQL
    mssql_cfg = {"engine": "mssql", "host": "sql.corp", "port": 1433, "database": "crm", "username": "sa", "password": "pwd"}
    assert "mssql+pyodbc://" in DBConfigManager.resolve_connection_url(mssql_cfg)
    assert "ODBC+Driver" in DBConfigManager.resolve_connection_url(mssql_cfg)

    # 5. SQLite
    sqlite_cfg = {"engine": "sqlite", "path": "data/app.db"}
    assert DBConfigManager.resolve_connection_url(sqlite_cfg) == "sqlite:///data/app.db"

    # 6. Direct URL override
    direct_cfg = {"url": "postgresql://custom_url_override"}
    assert DBConfigManager.resolve_connection_url(direct_cfg) == "postgresql://custom_url_override"


def test_json_config_loading(tmp_path):
    """Verifies that JSON configuration files are loaded with identical fidelity."""
    json_data = {
        "database": {
            "engine": "sqlite",
            "path": "tests/data/sample_warehouse.db",
            "table": "customers",
            "target": "churn"
        }
    }
    cfg_json = os.path.join(tmp_path, "db_config.json")
    with open(cfg_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f)

    config = DBConfigManager.load_config(cfg_json)
    assert config["engine"] == "sqlite"
    assert config["table"] == "customers"
    assert config["target"] == "churn"

    url = DBConfigManager.resolve_connection_url(config)
    assert url == "sqlite:///tests/data/sample_warehouse.db"


def test_password_masking_security():
    """Verifies that passwords are never exposed in log / UI summaries."""
    config = {
        "engine": "postgresql",
        "username": "admin",
        "password": "SuperSecretPassword!",
        "url": "postgresql://admin:SuperSecretPassword!@localhost:5432/dw"
    }
    masked = DBConfigManager.get_masked_summary(config)
    assert masked["password"] == "********"
    assert "SuperSecretPassword" not in masked["url"]
    assert ":********@" in masked["url"]


def test_sdk_execution_with_config_file(tmp_path):
    """Verifies that AutoDataAnalyzer SDK can execute seamlessly using a config file."""
    cfg_data = {
        "database": {
            "engine": "sqlite",
            "path": "tests/data/sample_warehouse.db",
            "table": "customers",
            "target": "churn"
        }
    }
    cfg_path = os.path.join(tmp_path, "sdk_config.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg_data, f)

    out_dir = os.path.join(tmp_path, "sdk_config_out")
    analyzer = AutoDataAnalyzer(output_dir=out_dir, sample_threshold=500)

    result = analyzer.analyze(config_path=cfg_path)
    assert result is not None
    assert "ml_scout" in result
    assert result["ml_scout"]["best_model"] is not None
    assert os.path.exists(os.path.join(out_dir, "customers_analysis_report.xlsx"))
