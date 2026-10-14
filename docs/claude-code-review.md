# Claude Code on GitHub

This repository uses [Claude Code](https://github.com/anthropics/claude-code-action)
to help review pull requests and respond to `@claude` mentions.

Two workflows are provided:

| Workflow | File | Trigger |
| --- | --- | --- |
| **Claude Code Review** | `.github/workflows/claude-code-review.yml` | Automatically on every PR (`opened`, `synchronize`) |
| **Claude Code** (interactive) | `.github/workflows/claude-code.yml` | When `@claude` is mentioned in an issue, PR comment, or review |

Authentication is via **Amazon Bedrock using GitHub OIDC** — no long-lived
Anthropic API key is stored in the repository.

## One-time setup (repo admin)

The workflows will no-op until an admin completes the steps below.

### 1. Create a GitHub OIDC identity provider in your AWS account

If your account doesn't already have one:

- Provider URL: `https://token.actions.githubusercontent.com`
- Audience: `sts.amazonaws.com`

### 2. Create an IAM role the workflows can assume

Create a role with a trust policy scoped to this repository. Example
(replace `<ACCOUNT_ID>` and confirm the `sub` scope matches your needs):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:aws-neuron/neuronx-distributed-inference:*"
        }
      }
    }
  ]
}
```

Attach a permissions policy allowing Bedrock model invocation:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "*"
    }
  ]
}
```

> Scope `Resource` to specific model/inference-profile ARNs if your security
> posture requires it. Some Claude models on Bedrock use cross-region inference
> profiles, so grant access in every region the profile spans.

### 3. Request Bedrock model access

In the Bedrock console (in the region you'll use), request access to the Claude
model you intend to run — by default the workflows use
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`.

### 4. Configure the repository secret and variables

Under **Settings → Secrets and variables → Actions**:

**Secret (required):**

- `CLAUDE_CODE_BEDROCK_ROLE_ARN` — ARN of the IAM role from step 2.

**Variables (optional — sensible defaults are built in):**

- `CLAUDE_CODE_AWS_REGION` — Bedrock region (default `us-west-2`).
- `CLAUDE_CODE_BEDROCK_MODEL` — Bedrock model id / inference-profile id
  (default `us.anthropic.claude-sonnet-4-5-20250929-v1:0`).

## Notes

- **Fork pull requests**: This is a public repository. PRs opened from forks do
  not receive OIDC credentials, so the auto-review job is a no-op for them. It
  runs for PRs from branches within this repository. This is the safe default —
  it avoids exposing credentials to untrusted PR code. To review fork PRs, a
  maintainer can `@claude` on the PR from a trusted context, or the workflow can
  later be extended with a label-gated `pull_request_target` trigger (review the
  [security implications](https://securitylab.github.com/resources/github-actions-preventing-pwn-requests/)
  first).
- **Cost control**: The interactive workflow only runs when a comment/issue
  contains `@claude`. The review workflow runs once per PR push.
- To disable a workflow without deleting it, disable it under the repo's
  **Actions** tab, or delete the corresponding file.
