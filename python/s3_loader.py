import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class S3ModelLoader:
    def __init__(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME', 'video-understanding-models')
        self.region = os.getenv('S3_REGION', 'us-east-1')
        self.cache_dir = Path(os.getenv('MODEL_CACHE_DIR', './python/models_cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        aws_access = os.getenv('S3_ACCESS_KEY')
        aws_secret = os.getenv('S3_SECRET_KEY')

        if aws_access and aws_secret:
            self.s3 = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=aws_access,
                aws_secret_access_key=aws_secret
            )
            self.use_s3 = True
        else:
            self.use_s3 = False
            print("[WARN] S3 credentials not found. Using local model cache only.")

    def get_model_path(self, model_filename: str) -> str:
        local_path = self.cache_dir / model_filename

        if local_path.exists():
            return str(local_path)

        if self.use_s3:
            print(f"[INFO] Downloading {model_filename} from S3...")
            self.s3.download_file(self.bucket_name, model_filename, str(local_path))
            print(f"[INFO] Downloaded {model_filename} to {local_path}")
            return str(local_path)

        raise FileNotFoundError(
            f"Model {model_filename} not found locally and S3 not available. "
            f"Please place the model in {self.cache_dir} or configure S3 credentials."
        )

    def ensure_all_models(self, model_list: list) -> dict:
        paths = {}
        for model_name in model_list:
            paths[model_name] = self.get_model_path(model_name)
        return paths
