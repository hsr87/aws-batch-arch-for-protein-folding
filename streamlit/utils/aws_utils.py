import boto3
from botocore.config import Config

def setup_aws_clients(region="us-west-2"):
    """AWS 클라이언트 설정"""
    bedrock_config = Config(
        connect_timeout=300,
        read_timeout=300,
        retries={"max_attempts": 10}
    )
    
    bedrock_rt = boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=bedrock_config
    )
    
    dynamodb = boto3.resource('dynamodb')
    
    return bedrock_rt, dynamodb