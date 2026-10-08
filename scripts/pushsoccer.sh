#!/usr/bin/env bash
set -euo pipefail

PERSONAL="$HOME/Repos/SoccerMon-Injury-Prediction"
CSUN_URL="https://github.com/CSU-Northridge-ARCS-Dev/Intelligent_Algorithms.git"
CSUN_FOLDER="Soccer_Injury_Forecasting"

cd "$PERSONAL"

# Verify the correct repository and branch.
test "$(git branch --show-current)" = "main" || {
    echo "ERROR: Personal repository must be on main."
    exit 1
}

# Do not silently omit uncommitted changes.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Uncommitted changes detected."
    echo "Commit your intended files before running pushsoccer."
    exit 1
fi

# Push personal repository.
echo "Pushing personal GitHub..."
git push origin main

# Create temporary CSUN checkout.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "Cloning CSUN repository..."
git clone --depth 1 --sparse "$CSUN_URL" "$TMP/csun"

cd "$TMP/csun"
git sparse-checkout set "$CSUN_FOLDER"

# Require the designated CSUN folder to exist.
test -d "$CSUN_FOLDER" || {
    echo "ERROR: CSUN destination folder not found."
    exit 1
}

# Ensure the destination contains no tracked files
# that are absent from the personal repository.
# Replace only the designated project folder.
# Refuse to overwrite unexpected research projects.
for project in     Physical_Activity_Classification     Runner_Injury_Forecasting     Sleep_Classification     Stress_Detection     Soccer_Injury_Forecasting
do
    if git ls-files -- "$CSUN_FOLDER/$project/" | grep -q .; then
        echo "ERROR: Unexpected project directory: $project"
        exit 1
    fi
done

git rm -r --ignore-unmatch -- "$CSUN_FOLDER"

mkdir -p "$CSUN_FOLDER"

# Export only committed personal files.
git -C "$PERSONAL" archive HEAD | tar -xf - -C "$CSUN_FOLDER"

git add -A -- "$CSUN_FOLDER"

# Skip commit when both copies are already identical.
if git diff --cached --quiet; then
    echo "CSUN already synchronized."
else
    echo "Committing CSUN synchronization..."
    git commit -m "Sync SoccerMon from personal repository"

    echo "Pushing CSUN GitHub..."
    git push origin main
fi

echo "Both GitHub repositories synchronized successfully."
