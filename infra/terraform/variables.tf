variable "name" {
  type    = string
  default = "forecastops"
}
variable "region" {
  type    = string
  default = "eu-west-1"
}
variable "vpc_id" { type = string }
variable "subnet_ids" {
  type        = list(string)
  description = "Reachable subnets, one per availability zone, with NAT or public-IP egress"
}
variable "allowed_cidrs" {
  type        = list(string)
  description = "Explicit trusted client CIDRs permitted to reach the demo API on port 8000"
}
variable "database_secret_arn" {
  type        = string
  description = "Secrets Manager secret containing the complete Postgres SQLAlchemy URL"
}
variable "tracking_uri" {
  type        = string
  description = "Network-reachable MLflow tracking server; no credentials in this value"
}
variable "image_tag" {
  type    = string
  default = "bootstrap"
}
variable "desired_count" {
  type    = number
  default = 0
  validation {
    condition     = contains([0, 1], var.desired_count)
    error_message = "V1 runs one API task; choose zero or one."
  }
}
variable "assign_public_ip" {
  type    = bool
  default = false
}
