import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Any, Dict

import PyPDF2
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType, EntityNode
from graphiti_core.edges import EntityEdge

from config import config

logger = logging.getLogger(__name__)


class MedicalKnowledgeGraph:
    def __init__(self):
        self.graphiti: Optional[Graphiti] = None
        
    async def initialize(self):
        try:
            self.graphiti = Graphiti(
                uri=config.neo4j.uri,
                user=config.neo4j.user,
                password=config.neo4j.password
            )
            
            await self.graphiti.build_indices_and_constraints()
            logger.info("Medical knowledge graph initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            raise
        
    async def close(self):
        if self.graphiti:
            try:
                await self.graphiti.close()
                logger.info("Medical knowledge graph connection closed")
            except Exception as e:
                logger.error(f"Error closing knowledge graph connection: {e}")
            
    async def ingest_static_document(self, file_path: Path, uploader: str = "system"):
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
            
        try:
            content = ""
            
            if file_path.suffix.lower() == '.pdf':
                content = self._extract_pdf_content(file_path)
            elif file_path.suffix.lower() == '.txt':
                content = file_path.read_text(encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
            if not content.strip():
                logger.warning(f"No content extracted from {file_path.name}")
                return
                
            
            await self.graphiti.add_episode(
                name=f"doc-{file_path.name}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                episode_body=content,
                source=EpisodeType.text,
                source_description=f"static:document:{uploader}",
                reference_time=datetime.now(timezone.utc),
            )
            
            logger.info(f"Successfully ingested document: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path.name}: {e}")
            raise
            
    def _extract_pdf_content(self, file_path: Path) -> str:
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                content = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    except Exception as e:
                        logger.warning(f"Could not extract text from page {page_num + 1}: {e}")
                        
                return content.strip()
                
        except Exception as e:
            logger.error(f"Error extracting PDF content from {file_path}: {e}")
            raise
            
    async def add_episode(self, name: str, episode_body: str, source: EpisodeType = EpisodeType.text, 
                         source_description: str = "user", reference_time: Optional[datetime] = None,
                         extra_props: Optional[Dict[str, Any]] = None, **kwargs):
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
            
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
            
        try:
            await self.graphiti.add_episode(
                name=name,
                episode_body=episode_body,
                source=source,
                source_description=source_description,
                reference_time=reference_time,
                **kwargs
            )
            
            logger.debug(f"Added episode: {name}")
            
        except Exception as e:
            logger.error(f"Error adding episode {name}: {e}")
            raise
            
    async def add_user_fact(self, user_id: str, raw_data: str):
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
        try:
            episode_body = raw_data.strip()
            source_type = EpisodeType.text
            try:
                parsed_json = json.loads(raw_data)
                source_type = EpisodeType.json
                episode_body = json.dumps(parsed_json, indent=2)
            except json.JSONDecodeError:
                pass
                
            episode_name = f"{user_id}-medical-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
            
            await self.graphiti.add_episode(
                name=episode_name,
                episode_body=episode_body,
                source=source_type,
                source_description=f"user:{user_id}:medical-data",
                reference_time=datetime.now(timezone.utc),
            )
            
            logger.info(f"Added medical data for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error adding user fact for {user_id}: {e}")
            raise
            
    async def get_nodes_by_query(self, query: str, limit: int = 10) -> List[EntityNode]:
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
        try:
            results = await self.graphiti.search_(query)
            return results.nodes[:limit]
            
        except Exception as e:
            logger.error(f"Error searching for nodes with query '{query}': {e}")
            raise
            
    async def search(self, query: str, center_node_uuid: Optional[str] = None, 
                    num_results: int = 5, group_ids: Optional[List[str]] = None) -> List[EntityEdge]:
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
            
        try:
            edges = await self.graphiti.search(
                query=query,
                center_node_uuid=center_node_uuid,
                num_results=num_results,
                group_ids=group_ids
            )
            
            logger.debug(f"Found {len(edges)} relevant facts for query: '{query}'")
            return edges
            
        except Exception as e:
            logger.error(f"Error searching knowledge graph with query '{query}': {e}")
            raise
            
    async def get_user_context(self, user_id: str, limit: int = 10) -> Dict[str, Any]:
        if not self.graphiti:
            raise RuntimeError("Knowledge graph not initialized")
            
        try:
            
            user_nodes = await self.get_nodes_by_query(user_id, limit=5)
            
            related_facts = []
            if user_nodes:
                user_node_uuid = user_nodes[0].uuid
                related_facts = await self.search(
                    query=user_id,
                    center_node_uuid=user_node_uuid,
                    num_results=limit
                )
            
            return {
                "user_nodes": user_nodes,
                "related_facts": related_facts,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error getting user context for {user_id}: {e}")
            raise
            
    async def health_check(self) -> bool:
        try:
            if not self.graphiti:
                return False
                
            
            await self.search("test", num_results=1)
            return True
            
        except Exception as e:
            logger.warning(f"Knowledge graph health check failed: {e}")
            return False


medical_kg = MedicalKnowledgeGraph()
