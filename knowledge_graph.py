import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Dict

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import SearchFilters

from config import config
from utils.pdf import extract_pdf_content

logger = logging.getLogger(__name__)


class MedicalKnowledgeGraph:
    def __init__(self):
        self.graphiti: Optional[Graphiti] = None
        
    async def initialize(self):
        try:
            logger.info("Initializing Graphiti connection...")
            logger.info(f"Neo4j URI: {config.neo4j.uri}")
            logger.info(f"Neo4j User: {config.neo4j.user}")
            
            self.graphiti = Graphiti(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password
            )
            
            logger.info("Building indices and constraints...")
            await self.graphiti.build_indices_and_constraints()
            logger.info("Medical knowledge graph initialized successfully")
            
        except Exception as e:
            error_msg = f"Failed to initialize knowledge graph: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Initialization traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
        
    async def close(self):
        if self.graphiti:
            try:
                await self.graphiti.close()
                logger.info("Medical knowledge graph connection closed")
            except Exception as e:
                error_msg = f"Error closing knowledge graph connection: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Close traceback: {traceback.format_exc()}")
    
    async def health_check(self) -> bool:
        try:
            if not self.graphiti:
                logger.warning("Health check failed: Graphiti not initialized")
                return False
                
            logger.debug("Performing health check...")
            await self.search("test", num_results=1)
            logger.debug("Health check passed")
            return True
            
        except Exception as e:
            error_msg = f"Knowledge graph health check failed: {type(e).__name__}: {str(e)}"
            logger.warning(error_msg)
            logger.debug(f"Health check traceback: {traceback.format_exc()}")
            return False
    
    async def search(
        self, 
        query: str, 
        center_node_uuid: Optional[str] = None,
        num_results: int = 10
    ) -> List[EntityEdge]:
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.debug(f"Searching knowledge graph with query: '{query}', center_node: {center_node_uuid}, limit: {num_results}")
            
            results = await self.graphiti.search(
                query=query,
                center_node_uuid=center_node_uuid,
                num_results=num_results
            )
            
            logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            error_msg = f"Error searching knowledge graph: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Search traceback: {traceback.format_exc()}")
            return []
    
    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: Optional[datetime] = None,
        source: EpisodeType = EpisodeType.text,
        group_id: str = '',
        extra_props: Optional[Dict[str, Any]] = None
    ):
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
            
        try:
            logger.debug(f"Adding episode '{name}' with source '{source_description}'")
            
            result = await self.graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source_description=source_description,
                reference_time=reference_time,
                source=source,
                group_id=group_id
            )
            
            logger.debug(f"Successfully added episode '{name}' to knowledge graph")
            return result
            
        except Exception as e:
            error_msg = f"Error adding episode '{name}' to knowledge graph: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Add episode traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    async def ingest_static_document(self, file_path: Path, uploader: str = "admin"):
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if file_path.suffix.lower() == '.pdf':
                logger.debug("Extracting PDF content...")
                content = extract_pdf_content(file_path)
            elif file_path.suffix.lower() == '.txt':
                logger.debug("Reading text content...")
                content = file_path.read_text(encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            if not content.strip():
                raise ValueError("Document contains no extractable text")
            
            logger.debug(f"Extracted {len(content)} characters from document")
            
            episode_name = f"doc-{file_path.stem}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            
            await self.add_episode(
                name=episode_name,
                episode_body=content,
                source_description=f"document:{uploader}:{file_path.name}",
                source=EpisodeType.text,
                group_id="static",
                reference_time=datetime.now(timezone.utc)
            )
            
            logger.info(f"Successfully ingested document: {file_path.name}")
            
        except Exception as e:
            error_msg = f"Error ingesting document {file_path}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Document ingestion traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    async def add_user_fact(self, user_id: str, fact_data: str):
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.debug(f"Adding user fact for user: {user_id}")
            
            if not user_id.strip():
                raise ValueError("User ID cannot be empty")
            if not fact_data.strip():
                raise ValueError("Fact data cannot be empty")
            
            episode_name = f"user-{user_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            
            await self.add_episode(
                name=episode_name,
                episode_body=fact_data,
                source_description=f"user-data:{user_id}",
                source=EpisodeType.text,
                group_id=f"user:{user_id}",
                reference_time=datetime.now(timezone.utc)
            )
            
            logger.info(f"Added medical data for user: {user_id}")
            
        except Exception as e:
            error_msg = f"Error adding user fact for {user_id}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Add user fact traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    async def get_nodes_by_query(self, query: str, limit: int = 10) -> List[EntityNode]:
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.debug(f"Getting nodes by query: '{query}' with limit: {limit}")
            
            search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            search_config.limit = limit
            
            results = await self.graphiti.search_(
                query=query,
                config=search_config,
                search_filter=SearchFilters()
            )
            
            logger.debug(f"Node search for '{query}' returned {len(results.nodes)} nodes")
            return results.nodes
            
        except Exception as e:
            error_msg = f"Error searching for nodes with query '{query}': {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Node search traceback: {traceback.format_exc()}")
            return []
    
    async def get_user_medical_history(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.graphiti:
            error_msg = "Knowledge graph not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        try:
            logger.debug(f"Getting medical history for user: {user_id}")
            
            group_ids = [f"user:{user_id}"]
            
            results = await self.graphiti.search(
                query=f"user {user_id} medical history",
                group_ids=group_ids,
                num_results=limit
            )
            
            history = []
            for edge in results:
                try:
                    history_item = {
                        "fact": edge.fact,
                        "created_at": edge.created_at.isoformat() if edge.created_at else None,
                        "valid_at": edge.valid_at.isoformat() if edge.valid_at else None,
                        "invalid_at": edge.invalid_at.isoformat() if edge.invalid_at else None,
                        "source": edge.name,
                        "episodes": edge.episodes
                    }
                    history.append(history_item)
                except Exception as item_error:
                    logger.warning(f"Error processing history item: {item_error}")
                    continue
            
            logger.debug(f"Retrieved {len(history)} medical history items for user {user_id}")
            return history
            
        except Exception as e:
            error_msg = f"Error retrieving medical history for {user_id}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Medical history traceback: {traceback.format_exc()}")
            return []


medical_kg = MedicalKnowledgeGraph()