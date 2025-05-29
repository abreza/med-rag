import os
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

@dataclass
class Neo4jConfig:
    uri: str
    user: str
    password: str

@dataclass
class OpenAIConfig:
    api_key: str
    model: str
    embedding_model: str

@dataclass
class AppConfig:
    host: str
    port: int
    share: bool
    debug: bool

@dataclass
class Config:
    neo4j: Neo4jConfig
    openai: OpenAIConfig
    app: AppConfig

def load_config() -> Config:
    required_vars = ['NEO4J_PASSWORD', 'OPENAI_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return Config(
        neo4j=Neo4jConfig(
            uri=os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.getenv('NEO4J_USER', 'neo4j'),
            password=os.getenv('NEO4J_PASSWORD')
        ),
        openai=OpenAIConfig(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            embedding_model=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small')
        ),
        app=AppConfig(
            host=os.getenv('APP_HOST', '0.0.0.0'),
            port=int(os.getenv('APP_PORT', 7860)),
            share=os.getenv('APP_SHARE', 'false').lower() == 'true',
            debug=os.getenv('DEBUG', 'false').lower() == 'true'
        ),
    )

config = load_config()