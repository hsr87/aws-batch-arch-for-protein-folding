"""애플리케이션 설정"""

# AWS 설정
AWS_CONFIG = {
    "region": "us-west-2",
    "model_name": "anthropic.claude-3-5-sonnet-20241022-v2:0",
}

# DynamoDB 설정
DYNAMODB_CONFIG = {
    "table_name": "Experiment",
}

# 파일 경로 설정
FILE_PATHS = {
    "molecule_dir": "molecule",
    "protein_dir": "protein",
}