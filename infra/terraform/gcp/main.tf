# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
  ])
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# Artifact Registry — Docker images
resource "google_artifact_registry_repository" "burst" {
  project       = var.project_id
  location      = var.region
  repository_id = "burst-classifier"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# Cloud Storage — model artifacts + MLflow backend
resource "google_storage_bucket" "models" {
  project                     = var.project_id
  name                        = var.bucket_name
  location                    = var.region
  uniform_bucket_level_access = true
  versioning { enabled = true }

  lifecycle_rule {
    condition { age = 90 }
    action { type = "Delete" }
  }
}

# Cloud Run — inference API
resource "google_cloud_run_v2_service" "api" {
  project  = var.project_id
  name     = "burst-classifier-api"
  location = var.region

  template {
    containers {
      # placeholder — Cloud Build overwrites this on first successful pipeline run
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      env {
        name  = "GCS_BUCKET"
        value = var.bucket_name
      }
      env {
        name  = "MODEL_PATH"
        value = "models/latest/model.onnx"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }

    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  depends_on = [google_project_service.apis]
}

# Public access set via gcloud (requires run.services.setIamPolicy)

data "google_project" "project" {
  project_id = var.project_id
}

# NOTE: Cloud Build SA IAM grants and Cloud Run public access are applied
# via gcloud commands (see outputs.tf instructions) because they require
# resourcemanager.projectIamAdmin which is managed outside Terraform.
