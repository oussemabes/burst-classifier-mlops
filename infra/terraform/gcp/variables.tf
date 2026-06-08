variable "project_id" {
  default = "kwore-web-dev"
}

variable "region" {
  default = "europe-west1"
}

variable "credentials_file" {
  description = "Path to GCP service account JSON key"
  default     = "../../../kwore-web-dev-ba03a9222e09.json"
}

variable "bucket_name" {
  default = "kwore-web-dev-burst-classifier"
}

variable "image_name" {
  default = "burst-classifier-api"
}
