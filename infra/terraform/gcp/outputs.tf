output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/burst-classifier"
}

output "gcs_bucket" {
  value = google_storage_bucket.models.name
}

output "cloud_run_url" {
  value = google_cloud_run_v2_service.api.uri
}
