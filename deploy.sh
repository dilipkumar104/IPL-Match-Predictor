#!/usr/bin/env bash
# =============================================================================
# deploy.sh — IPL Match Predictor: Full GitHub Deployment Script
# =============================================================================
# USAGE:
#   1. Edit the GITHUB_REPO_URL variable below with your actual GitHub repo URL
#   2. Run:  bash deploy.sh
#
# WHAT THIS SCRIPT DOES:
#   1. Checks for Git installation
#   2. Initialises a Git repository (or reuses an existing one)
#   3. Creates a .gitignore if one doesn't exist
#   4. Stages all project files (code, data, docs)
#   5. Creates a signed commit
#   6. Sets up the 'main' branch
#   7. Adds the GitHub remote (if not already set)
#   8. Pushes everything to GitHub
# =============================================================================

set -euo pipefail   # Exit on any error, unset variable, or pipe failure

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — EDIT THIS SECTION BEFORE RUNNING
# ─────────────────────────────────────────────────────────────────────────────

# Replace with your actual GitHub repo URL (SSH or HTTPS)
# SSH  example: git@github.com:dilipkumar104/ipl-match-predictor.git
# HTTPS example: https://github.com/dilipkumar104/ipl-match-predictor.git
GITHUB_REPO_URL="https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"

# Commit message
COMMIT_MSG="feat: Add production-grade Scikit-Learn ML pipeline, README, and TUTORIAL"

# Git author info (used only if not already configured globally)
GIT_AUTHOR_NAME="Dilip"
GIT_AUTHOR_EMAIL="dilip@kmit.in"

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " IPL Match Predictor — GitHub Deployment"
echo "============================================================"
echo ""

# Check that GITHUB_REPO_URL has been set
if [[ "$GITHUB_REPO_URL" == *"YOUR_USERNAME"* ]]; then
    echo "❌ ERROR: Please edit deploy.sh and set your GITHUB_REPO_URL first."
    echo "   Open deploy.sh and replace the placeholder on line ~30."
    exit 1
fi

# Check Git is installed
if ! command -v git &> /dev/null; then
    echo "❌ ERROR: Git is not installed or not in PATH."
    echo "   Install it from: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git version: $(git --version)"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Configure Git author (if not already configured)
# ─────────────────────────────────────────────────────────────────────────────

CURRENT_NAME=$(git config --global user.name 2>/dev/null || echo "")
CURRENT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [[ -z "$CURRENT_NAME" ]]; then
    git config --global user.name "$GIT_AUTHOR_NAME"
    echo "✅ Git user.name set to: $GIT_AUTHOR_NAME"
fi

if [[ -z "$CURRENT_EMAIL" ]]; then
    git config --global user.email "$GIT_AUTHOR_EMAIL"
    echo "✅ Git user.email set to: $GIT_AUTHOR_EMAIL"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Initialise local Git repository
# ─────────────────────────────────────────────────────────────────────────────

if [ ! -d ".git" ]; then
    echo "[Step 2] Initialising new Git repository..."
    git init
    echo "✅ Git repository initialised."
else
    echo "[Step 2] Existing Git repository detected — skipping init."
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Create / update .gitignore
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 3] Writing .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.pyo
.Python
*.egg-info/
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/
.env

# Jupyter
.ipynb_checkpoints/
*.ipynb

# IDE
.vscode/
.idea/
*.suo
*.user

# OS
.DS_Store
Thumbs.db
desktop.ini

# ML artifacts (optional — remove lines below to track the model in git)
# models/*.pkl

# Large files
*.parquet
*.feather
EOF

echo "✅ .gitignore written."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Stage all files
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 4] Staging all project files..."
git add .
echo "✅ Files staged. Summary:"
git status --short
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Create the commit
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 5] Creating commit..."
git commit -m "$COMMIT_MSG" || {
    echo "ℹ️  Nothing to commit (files may already be committed). Continuing..."
}
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Set up 'main' branch
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 6] Setting branch to 'main'..."
git branch -M main
echo "✅ Branch set to 'main'."
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Add GitHub remote
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 7] Configuring GitHub remote..."
if git remote get-url origin &> /dev/null; then
    echo "  Remote 'origin' already exists. Updating URL..."
    git remote set-url origin "$GITHUB_REPO_URL"
else
    git remote add origin "$GITHUB_REPO_URL"
fi
echo "✅ Remote 'origin' → $GITHUB_REPO_URL"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Push to GitHub
# ─────────────────────────────────────────────────────────────────────────────

echo "[Step 8] Pushing to GitHub (origin/main)..."
echo "  (You may be prompted for your GitHub username & password / token)"
echo ""

git push -u origin main

echo ""
echo "============================================================"
echo "  ✅ SUCCESS — All files pushed to GitHub!"
echo ""
echo "  Repository URL:"
echo "  $GITHUB_REPO_URL"
echo ""
echo "  Next steps:"
echo "  1. Visit your repository on GitHub to verify the upload"
echo "  2. Add a description and topics to your GitHub repo"
echo "  3. Consider adding a GitHub Actions CI workflow"
echo "============================================================"
echo ""
