#!/bin/bash
# Push changes to GitHub

echo "📦 Preparing to push to GitHub..."

# Configure git if needed (generic)
git config --global user.email "swelu-agent@example.com"
git config --global user.name "SWELU Agent"

# Add all changes
git add .

# Commit
git commit -m "Update: Hyper SWELU Model (Paul OBARA Logic) + Kelly-Taguchi LR + Streaming"

# Push (assuming remote 'origin' is set)
echo "🚀 Pushing to origin..."
git push origin main

echo "✅ Done!"
