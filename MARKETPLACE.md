# Marketplace Publishing Checklist (`token-reducer`)

## 1) Create public GitHub repository

- Repository: `https://github.com/Madhan230205/token-reducer`
- Push this folder as repository root.

## 2) Register repository as a Claude plugin marketplace

- In Claude Code:
  - `/plugin marketplace add Madhan230205/token-reducer`

## 3) Install and verify

- Install:
  - `/plugin install token-reducer`
- Verify plugin is active:
  - `/plugin list`
- Reload if needed:
  - `/reload-plugins`

## 4) Plug-and-play usage

- Use command:
  - `/token-reducer <your task question>`

## 5) Team rollout (optional)

- Project scope install:
  - `claude plugin install token-reducer@madhan230205-marketplace --scope project`
