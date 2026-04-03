## Overview
Repository metadata is defined in .github/settings.yml, but there is no guaranteed automation that applies this config to live GitHub repository settings.

## User Story
As an open-source maintainer
I want repository description/homepage/topics to sync automatically from versioned config
So that discoverability metadata is always consistent and auditable.

## Context
- Why needed: Declarative metadata is currently not guaranteed to be enforced.
- Current workflow: Maintainers edit .github/settings.yml manually.
- Pain point: GitHub UI can drift from repository config.
- Success metric: 100% successful metadata sync on pushes that change .github/settings.yml.

## Acceptance Criteria
- [ ] A workflow runs when .github/settings.yml changes.
- [ ] Repo description, homepage, and topics are updated via GitHub API.
- [ ] Workflow logs the final applied values.
- [ ] Failure mode is explicit when required permissions/secrets are missing.
- [ ] Contributing docs explain setup and fallback.

## Technical Requirements
- Technology/framework: GitHub Actions + gh api or actions/github-script
- Security: least-privileged token with required repository admin scope
- Performance: workflow completes in under 60s

## Definition of Done
- [ ] Workflow implemented and merged
- [ ] Docs updated
- [ ] Verified by changing description/topics in one PR and confirming GitHub UI sync

## Estimated Effort
1-2 days
