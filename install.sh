#!/usr/bin/env bash
pip install -e '.[dev]'
cd .git/hooks/ && ln -fs ../../config/hooks/pre-commit . && cd -
cd .git/hooks/ && ln -fs ../../config/hooks/pre-push . && cd -
