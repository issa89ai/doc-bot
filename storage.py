"""Optional cloud backups. On EC2 use an instance role, not static AWS keys."""
import os


def client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-2"),
                        config=Config(connect_timeout=5, read_timeout=20,
                                      retries={"max_attempts": 2}))


def backup_pdf(path, filename):
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        return "disabled"
    client().upload_file(path, bucket, f"pdfs/{filename}")
    return "saved"


def delete_backup(filename):
    bucket = os.getenv("S3_BUCKET")
    if bucket:
        client().delete_object(Bucket=bucket, Key=f"pdfs/{filename}")
