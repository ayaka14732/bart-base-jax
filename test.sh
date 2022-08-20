#!/bin/sh

for f in test/**/*.py; do
  echo 🍜 Testing "$f"...
  python "$f"
  echo
done
