#!/bin/bash
# GitHub Repository Setup Script for Snooker RL
# Run this script in your terminal after making sure git is installed

echo "=================================="
echo "Snooker RL - GitHub Setup"
echo "=================================="

# Navigate to project directory
cd /workspace/snooker_rl

# Configure git (replace with your info)
echo "Configuring git..."
git config --global user.name "Your GitHub Username"
git config --global user.email "your.email@example.com"

# Rename branch to main
git branch -M main

# Add remote (replace with your repo URL after creating it)
echo "Adding remote origin..."
echo "Please create a new repository on GitHub first at:"
echo "  https://github.com/new"
echo ""
echo "Then enter the repository URL (e.g., https://github.com/username/snooker_rl.git):"
read -p "Repository URL: " repo_url

git remote add origin "$repo_url"

# Stage all files
echo "Staging files..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Snooker RL with PPO implementation

- Physics-based snooker environment using PyMunk
- PPO (Proximal Policy Optimization) algorithm
- Training and evaluation scripts
- Comprehensive test suite"

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "=================================="
echo "Setup complete!"
echo "Repository URL: $repo_url"
echo "=================================="
