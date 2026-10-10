output "repository_url" { value = aws_ecr_repository.app.repository_url }
output "artifact_bucket" { value = aws_s3_bucket.artifacts.bucket }
output "cluster_name" { value = aws_ecs_cluster.app.name }
output "service_name" { value = aws_ecs_service.app.name }
output "task_definition_arn" { value = aws_ecs_task_definition.app.arn }
output "task_security_group" { value = aws_security_group.app.id }
