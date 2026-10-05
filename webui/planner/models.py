from django.db import models


class PlanRun(models.Model):
    class Status(models.TextChoices):
        RUNNING = "running", "Running"
        DONE = "done", "Completed"
        FAILED = "failed", "Failed"

    name = models.CharField(max_length=120, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.RUNNING)
    params = models.JSONField(default=dict)
    command = models.TextField(blank=True)
    output_prefix = models.CharField(max_length=255)
    log = models.TextField(blank=True)
    error = models.TextField(blank=True)
    summary = models.JSONField(default=dict)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name or f"Run {self.pk}"

    @property
    def display_name(self) -> str:
        return self.name or f"Run #{self.pk}"
