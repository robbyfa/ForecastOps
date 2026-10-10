# AWS deployment configuration

This module passes `terraform validate` with Terraform 1.10.5 and the locked AWS provider. It has not been applied to an AWS account. Configuration validation is not deployment evidence. The schema follows the official [AWS provider ECS task definition documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/ecs_task_definition).

Resources: immutable-tag ECR repository, encrypted/versioned private S3 bucket, ECS cluster/service/task, encrypted shared EFS volume/access point, restricted security groups, task/execution roles and CloudWatch logs. Shared EFS keeps the champion pointer and local artifacts available to API and batch tasks. MLflow must use a separately managed server/database; do not put its SQLite backend on EFS.

Supply existing VPC/subnets (one subnet per AZ), a reachable Postgres service, a Secrets Manager secret containing its full SQLAlchemy URL, and a network-reachable MLflow server. The application needs network access to the database, MLflow, ECR, CloudWatch and Secrets Manager. Configure NAT/VPC endpoints or public-IP routing accordingly. Allow Postgres ingress from the task security group; this module does not modify an external database's security group. Customer-managed secret encryption keys additionally require `kms:Decrypt` permissions, not included here.

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
# Replace every placeholder and configure your AWS identity.
terraform init
terraform fmt -check
terraform validate
terraform plan -out=review.tfplan
# Review cost, networking, IAM and all planned changes before applying.
terraform apply review.tfplan
```

Configure a remote encrypted state backend with locking before team use. The default service count is zero so the initial `bootstrap` image tag does not launch a missing image. EFS and other resources can still incur charges after apply. Publish the first application image, update its task definition, then set `desired_count = 1` and apply a reviewed plan. For a public-IP demo, `allowed_cidrs` should contain only trusted client addresses. For production, use private tasks with an authenticated TLS ingress/load balancer; that ingress is not supplied by this module.

The manual `Deploy AWS` workflow runs only from main and uses a GitHub `production` environment. Set these environment variables in GitHub:

- `AWS_DEPLOY_ROLE_ARN`, `AWS_REGION`
- `ECR_REPOSITORY` (repository name, e.g. `forecastops`)
- `ECS_TASK_FAMILY`, `ECS_CLUSTER`, `ECS_SERVICE`

Create a GitHub OIDC deployment role restricted to this repository and environment, with scoped ECR push, ECS register/update/describe and IAM pass-role permissions. The module creates application roles, not your deployment identity. Apply environment protection rules in GitHub as needed. Image tags include commit SHA, workflow run ID and attempt to support immutable retries. The service ignores Terraform task-definition changes after bootstrap because application releases are controlled by that workflow; infrastructure task-definition changes must also be rolled out deliberately.

Run batch work as a one-off Fargate task using the same task definition/network/EFS, overriding the container command with `forecastops train --data /app/artifacts/sales.csv` or `forecastops cycle --data /app/artifacts/sales.csv`. Override/disable its API health check for long-running training jobs. Dataset ingestion to shared storage and scheduling are explicit deployment tasks. The S3 bucket is available for datasets and report backups; V1 does not automatically mirror local artifacts to it.

Deployment acceptance requires a successful data import, registered model, batch forecast, Postgres/API round trip, CloudWatch logs, monitored actuals and a rollback exercise. Preserve that evidence before describing this as a deployed cloud platform.
