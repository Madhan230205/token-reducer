## Overview
GitHub release creation should be idempotent and robust across reruns and minor CLI behavior differences.

## User Story
As a maintainer
I want release creation to be resilient and rerunnable
So that tagged releases do not require manual intervention after failures.

## Context
- Current workflow creates release and skips if already exists.
- Pain point: reruns may not update/re-upload assets cleanly.
- Success metric: release pipeline succeeds on first run and reruns with no manual fixes.

## Acceptance Criteria
- [ ] New tag creates release with generated notes + intro text.
- [ ] Existing release path uploads artifacts with overwrite/clobber behavior.
- [ ] Logs clearly show whether release was created or updated.
- [ ] Release step remains compatible with default ubuntu-latest gh CLI.

## Technical Requirements
- Technology/framework: GitHub Actions + GitHub CLI
- Security: use GITHUB_TOKEN with contents:write only
- Performance: no major runtime increase

## Definition of Done
- [ ] Workflow updated
- [ ] Validated with one new tag and one rerun scenario
- [ ] Docs updated in contribute.md

## Estimated Effort
1 day
