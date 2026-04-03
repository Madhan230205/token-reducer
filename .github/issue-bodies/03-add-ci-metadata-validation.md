## Overview
Add CI checks to validate metadata/release automation prerequisites before maintainers cut tags.

## User Story
As a maintainer
I want CI to validate release metadata files
So that configuration mistakes are caught before publishing.

## Context
- Current checks validate version alignment, but not all metadata prerequisites.
- Pain point: late failures or silent metadata drift.
- Success metric: CI fails fast with actionable errors when required metadata files are invalid.

## Acceptance Criteria
- [ ] CI validates .github/settings.yml exists and has required keys.
- [ ] CI validates .github/release-intro.md exists and is non-empty.
- [ ] Error messages are explicit and actionable.
- [ ] Checks run on pull requests targeting main.

## Technical Requirements
- Technology/framework: lightweight script in CI (Python/yq/grep-based)
- Security: no elevated permissions
- Performance: <30s added CI time

## Definition of Done
- [ ] Validation step added to CI workflow
- [ ] Docs updated
- [ ] Demonstrated pass/fail behavior in PR

## Estimated Effort
1 day
