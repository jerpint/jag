#!/usr/bin/env bash
pip install -e '.[dev]'
cd .git/hooks/ && ln -fs ../../config/hooks/pre-commit . && cd -
chmod u+x .git/hooks/pre-commit
cd .git/hooks/ && ln -fs ../../config/hooks/pre-push . && cd -
chmod u+x .git/hooks/pre-push

