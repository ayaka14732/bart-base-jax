#!/bin/sh
set -e

for f in test/**/*.py; do
  echo 🍜 Testing "$f"...
  ./startpod python "$f"
  echo
done
