# Marketplace Publishing Checklist (`token-reducer`)

## 1) Create public GitHub repository

- Repository: `https://github.com/Madv6/token-reducer`
- Push this folder as repository root.

## 2) Register repository as a Claude plugin marketplace

- In Claude Code:
  - `/plugin marketplace add Madv6/token-reducer`

## 3) Install and verify

- Install:
  - `claude plugin install token-reducer@madv6-marketplace`
- Verify plugin is active:
  - `/plugin list`
- Reload if needed:
  - `/reload-plugins`

## 4) Plug-and-play usage

- Use command:
  - `/token-reducer <your task question>`

## 5) Team rollout (optional)

- Project scope install:
  - `claude plugin install token-reducer@madv6-marketplace --scope project`
