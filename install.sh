#!/usr/bin/env bash
pip install -e '.[dev]'
ln -sf ./config/hooks/pre-commit .git/hooks/
ln -sf ./config/hooks/pre-push .git/hooks/
