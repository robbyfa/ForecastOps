terraform {
  required_version = ">= 1.10, < 2.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = { Project = "ForecastOps", ManagedBy = "Terraform" }
  }
}

data "aws_caller_identity" "current" {}

resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_s3_bucket" "artifacts" {
  bucket = "${var.name}-${data.aws_caller_identity.current.account_id}-${var.region}"
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.name}"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "app" {
  name = var.name
}

resource "aws_security_group" "app" {
  name   = "${var.name}-app"
  vpc_id = var.vpc_id
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "efs" {
  name   = "${var.name}-efs"
  vpc_id = var.vpc_id
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.app.id]
  }
}

resource "aws_efs_file_system" "artifacts" {
  encrypted = true
}

resource "aws_efs_access_point" "artifacts" {
  file_system_id = aws_efs_file_system.artifacts.id
  posix_user {
    uid = 10001
    gid = 10001
  }
  root_directory {
    path = "/forecastops"
    creation_info {
      owner_uid   = 10001
      owner_gid   = 10001
      permissions = "0750"
    }
  }
}

resource "aws_efs_mount_target" "artifacts" {
  for_each        = toset(var.subnet_ids)
  file_system_id  = aws_efs_file_system.artifacts.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

locals {
  assume_task = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = "sts:AssumeRole",
    Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
}

resource "aws_iam_role" "execution" {
  name               = "${var.name}-execution"
  assume_role_policy = local.assume_task
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "database_secret" {
  role = aws_iam_role.execution.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [{
    Effect   = "Allow", Action = ["secretsmanager:GetSecretValue"],
    Resource = var.database_secret_arn
  }] })
}

resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = local.assume_task
}

resource "aws_iam_role_policy" "artifacts" {
  role = aws_iam_role.task.id
  policy = jsonencode({ Version = "2012-10-17", Statement = [
    { Effect   = "Allow", Action = ["elasticfilesystem:ClientMount", "elasticfilesystem:ClientWrite"],
      Resource = aws_efs_file_system.artifacts.arn,
    Condition = { StringEquals = { "elasticfilesystem:AccessPointArn" = aws_efs_access_point.artifacts.arn } } },
    { Effect = "Allow", Action = ["s3:ListBucket"], Resource = aws_s3_bucket.artifacts.arn },
    { Effect = "Allow", Action = ["s3:GetObject", "s3:PutObject"], Resource = "${aws_s3_bucket.artifacts.arn}/*" }
  ] })
}

resource "aws_ecs_task_definition" "app" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn
  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }
  volume {
    name = "artifacts"
    efs_volume_configuration {
      file_system_id     = aws_efs_file_system.artifacts.id
      transit_encryption = "ENABLED"
      authorization_config {
        access_point_id = aws_efs_access_point.artifacts.id
        iam             = "ENABLED"
      }
    }
  }
  container_definitions = jsonencode([{
    name         = "api", image = "${aws_ecr_repository.app.repository_url}:${var.image_tag}", essential = true,
    portMappings = [{ containerPort = 8000, protocol = "tcp" }],
    mountPoints  = [{ sourceVolume = "artifacts", containerPath = "/app/artifacts", readOnly = false }],
    environment = [
      { name = "FORECASTOPS_ARTIFACT_DIR", value = "/app/artifacts" },
      { name = "FORECASTOPS_TRACKING_URI", value = var.tracking_uri }
    ],
    secrets = [{ name = "FORECASTOPS_DATABASE_URL", valueFrom = var.database_secret_arn }],
    healthCheck = { command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
    interval = 30, timeout = 5, retries = 3, startPeriod = 60 },
    logConfiguration = { logDriver = "awslogs", options = {
      "awslogs-group"  = aws_cloudwatch_log_group.app.name,
      "awslogs-region" = var.region, "awslogs-stream-prefix" = "api"
    } }
  }])
}

resource "aws_ecs_service" "app" {
  name            = var.name
  cluster         = aws_ecs_cluster.app.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = [aws_security_group.app.id]
    assign_public_ip = var.assign_public_ip
  }
  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }
  depends_on = [aws_efs_mount_target.artifacts, aws_iam_role_policy_attachment.execution,
  aws_iam_role_policy.database_secret, aws_iam_role_policy.artifacts]
  lifecycle {
    # Image releases are handled by the deployment workflow after initial provisioning.
    ignore_changes = [task_definition]
  }
}
