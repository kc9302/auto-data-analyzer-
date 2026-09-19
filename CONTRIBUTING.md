# Contributing to Auto Data Analyzer

First off, thank you for considering contributing to **Auto Data Analyzer & ML Scout**! 🎉

We welcome contributions from everyone. Whether it's reporting a bug, improving documentation, writing a new data connector, or adding an algorithm to MLScout, your help is appreciated.

---

## 🧭 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for everyone. Please be respectful and considerate in all communications.

---

## 🛠️ Getting Started with Local Development

### 1. Prerequisites
- Python >= 3.10 (Python 3.12 recommended)
- `uv` package manager (recommended: `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Fork & Setup
```bash
# 1. Fork repository on GitHub, then clone your fork:
git clone https://github.com/<your-username>/auto-data-analyzer.git
cd auto-data-analyzer

# 2. Install dependencies with uv
uv sync

# 3. Verify tests pass
uv run pytest -v
```

---

## 🌿 Branching & PR Workflow

We follow the standard Gitflow / Feature-Branch workflow:

1. Create a branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feat/your-feature-name
   # or fix/your-bug-fix
   ```
2. Make clean, atomic commits:
   ```bash
   git commit -m "feat(connector): add Snowflake read-only query support"
   ```
3. Run tests and static analysis:
   ```bash
   uv run pytest -v
   ```
4. Push to your fork and submit a Pull Request targeting the `develop` branch.
5. In your PR description, explain the problem solved, proposed changes, and how you verified them.

---

## 💡 Good First Issues & Focus Areas

If you want to contribute, here are great places to start:
- 🔌 **Connectors**: Add read-only support for BigQuery, ClickHouse, or DuckDB.
- 📊 **Visualizations**: Improve executive PPTX templates or HTML dashboard styling.
- 🧪 **Metrics**: Add domain-specific classification or regression metrics to MLScout.
- 📖 **Documentation**: Translate docs, fix typos, or write tutorials for common database setups.

---

## 💬 Community & Questions

- **GitHub Issues**: For bug reports and feature requests.
- **GitHub Discussions**: For Q&A, roadmap discussions, and architecture brainstorming.
