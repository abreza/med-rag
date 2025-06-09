from __future__ import annotations

import asyncio
import json
import logging
import threading
import queue
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Callable
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from openai import AsyncOpenAI, OpenAIError

from config import config
from knowledge_graph import medical_kg
from graphiti_core.nodes import EpisodeType

logger = logging.getLogger(__name__)
openai_client = AsyncOpenAI(api_key=config.openai.api_key)


class AsyncOperationManager:
    def __init__(self):
        self.loop = None
        self.thread = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._start_loop()
    
    def _start_loop(self):
        self.result_queue = queue.Queue()
        
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            try:
                self.loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}")
                logger.error(f"Event loop traceback: {traceback.format_exc()}")
            finally:
                self.loop.close()
        
        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()
        
        import time
        while self.loop is None:
            time.sleep(0.01)
    
    def run_async(self, coro) -> Any:
        if self.loop is None or self.loop.is_closed():
            error_msg = "Event loop is not available"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result(timeout=300)
        except asyncio.TimeoutError as e:
            error_msg = f"Async operation timed out after 30 seconds"
            logger.error(error_msg)
            logger.error(f"Timeout traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Async operation failed: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def shutdown(self):
        try:
            if self.loop and not self.loop.is_closed():
                self.loop.call_soon_threadsafe(self.loop.stop)
            if self.thread:
                self.thread.join(timeout=5)
            self.executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            logger.error(f"Shutdown traceback: {traceback.format_exc()}")


async_manager = AsyncOperationManager()


def run_async(coro):
    try:
        return async_manager.run_async(coro)
    except Exception as e:
        error_msg = f"Error running async operation: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


async def _ingest_admin_inputs_async(files: List[Path], facts: str) -> str:
    try:
        if not files and not facts.strip():
            return "⚠️ Please upload at least one file or type some facts."

        messages = []
        if files:
            for f in files:
                await medical_kg.ingest_static_document(f, uploader="admin")
            messages.append(f"Ingested {len(files)} document(s)")

        facts_text = facts.strip()
        if facts_text:
            await medical_kg.add_episode(
                name="admin-facts-" + datetime.now(timezone.utc).isoformat(),
                episode_body=facts_text,
                source=EpisodeType.text,
                source_description="static:admin-facts",
                extra_props={"static": True},
            )
            messages.append("Ingested typed facts into the static knowledge graph")

        return "✅ " + "; ".join(messages)
    except Exception as e:
        error_msg = f"Error in _ingest_admin_inputs_async: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def ingest_admin_inputs(files: List[Path], facts: str) -> str:
    try:
        return run_async(_ingest_admin_inputs_async(files, facts))
    except Exception as e:
        error_msg = f"Error in ingest_admin_inputs: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"❌ Error: {error_msg}"


async def _add_user_medical_data_async(user_id: str, raw: str) -> str:
    try:
        user_id = user_id.strip()
        if not user_id:
            return "⚠️ User ID is required."
        if not raw.strip():
            return "⚠️ Medical data field is empty."

        await medical_kg.add_user_fact(user_id, raw)
        return "✅ Medical data saved to graph memory."
    except Exception as e:
        error_msg = f"Error in _add_user_medical_data_async: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def add_user_medical_data(user_id: str, raw: str) -> str:
    try:
        return run_async(_add_user_medical_data_async(user_id, raw))
    except Exception as e:
        error_msg = f"Error in add_user_medical_data: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return f"❌ Error: {error_msg}"


async def _get_or_create_user_node_async(user_id: str) -> str:
    try:
        matches = await medical_kg.get_nodes_by_query(user_id)
        if matches:
            return matches[0].uuid

        await medical_kg.add_episode(
            name=user_id,
            episode_body=f"Identifier node for user {user_id}",
            source_description="system:user-node",
            reference_time=datetime.now(timezone.utc),
        )
        matches = await medical_kg.get_nodes_by_query(user_id)
        if not matches:
            raise RuntimeError(f"Failed to create or retrieve user node for {user_id}")
        return matches[0].uuid
    except Exception as e:
        error_msg = f"Error in _get_or_create_user_node_async: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


async def _chat_bot_async(message: str, history: List[Dict[str, str]], user_id: str):
    try:
        history = history or []
        user_id = user_id.strip() or "anonymous"

        logger.info(f"Processing chat message for user {user_id}: {message[:100]}...")

        if not medical_kg.graphiti:
            error_msg = "Medical knowledge graph is not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        user_node_uuid = await _get_or_create_user_node_async(user_id)
        logger.debug(f"User node UUID: {user_node_uuid}")

        user_message_count = len([msg for msg in history if msg.get("role") == "user"])
        
        await medical_kg.add_episode(
            name=f"{user_id}-msg-{user_message_count}",
            episode_body=message,
            source_description="chat:user",
            reference_time=datetime.now(timezone.utc),
        )

        logger.debug(f"Searching knowledge graph for: {message}")
        hits = await medical_kg.search(message, center_node_uuid=user_node_uuid, num_results=5)
        logger.debug(f"Found {len(hits)} search results")
        
        facts_list: List[str] = []
        for h in hits:
            snippet = getattr(h, "fact", None) or getattr(h, "text", None) or str(h)
            facts_list.append(snippet.strip())
        facts = "\n".join(facts_list) or "(no matching graph facts)"

        system_prompt = (
            "You are MedAI, a cautious, evidence-based virtual medical assistant. "
            "Leverage the structured facts below plus standard clinical knowledge. "
            "If uncertain, ask clarifying questions and encourage consulting a professional.\n\n"
            f"### Facts from Knowledge Graph:\n{facts}"
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        logger.debug("Making OpenAI API call...")
        try:
            completion = await openai_client.chat.completions.create(
                model=config.openai.model,
                messages=messages,
                temperature=0.3,
            )
            assistant_reply = completion.choices[0].message.content
            logger.debug("OpenAI API call successful")
        except OpenAIError as err:
            assistant_reply = f"⚠️ OpenAI error: {err}"
            logger.error(f"OpenAI error: {err}")
            logger.error(f"OpenAI error traceback: {traceback.format_exc()}")

        await medical_kg.add_episode(
            name=f"{user_id}-assistant-{user_message_count}",
            episode_body=assistant_reply,
            source_description="chat:assistant",
            reference_time=datetime.now(timezone.utc),
        )

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": assistant_reply})
        logger.info(f"Chat response generated successfully for user {user_id}")
        return history
        
    except Exception as e:
        error_msg = f"Error in _chat_bot_async: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise


def chat_bot(message: str, history: List[Dict[str, str]], user_id: str):
    try:
        return run_async(_chat_bot_async(message, history, user_id))
    except Exception as e:
        error_msg = f"Error in chat_bot: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        error_history = history or []
        error_history.append({"role": "user", "content": message})
        error_history.append({
            "role": "assistant", 
            "content": f"❌ Error: {error_msg}\n\nPlease check the logs for more details."
        })
        return error_history


def clear_message():
    return ""


def clear_chat():
    return []


def create_interface():
    with gr.Blocks(title="Medical Knowledge-Graph Assistant") as demo:
        gr.Markdown("# 🩺 Medical Knowledge-Graph Assistant")

        with gr.Tab("Admin Upload"):
            gr.Markdown("### Upload Medical Documents and Facts")
            gr.Markdown("Use this tab to upload medical documents or add general medical facts to the knowledge graph.")
            
            file_box = gr.Files(
                file_types=[".pdf", ".txt"],
                label="Upload medical documents (PDF/TXT)",
            )
            facts_area = gr.TextArea(
                label="Type medical facts",
                lines=5,
                placeholder="Enter facts or observations here..."
            )
            ingest_btn = gr.Button("Ingest to Graph", variant="primary")
            ingest_status = gr.Markdown()
            
            ingest_btn.click(
                ingest_admin_inputs,
                inputs=[file_box, facts_area],
                outputs=ingest_status,
            )

        with gr.Tab("Add Medical Data"):
            gr.Markdown("### Add Patient-Specific Medical Data")
            gr.Markdown("Use this tab to add medical data for specific patients/users.")
            
            user_id_text = gr.Text(label="User ID", placeholder="Enter patient/user identifier")
            med_data_area = gr.TextArea(
                label="Medical Data (JSON or text)", 
                lines=10,
                placeholder="Enter medical data in JSON format or as plain text..."
            )
            add_btn = gr.Button("Save to Memory", variant="primary")
            add_status = gr.Markdown()
            
            add_btn.click(
                add_user_medical_data,
                inputs=[user_id_text, med_data_area],
                outputs=add_status,
            )

        with gr.Tab("Chat"):
            gr.Markdown("### Medical Assistant Chat")
            gr.Markdown("Chat with the AI assistant that has access to the medical knowledge graph.")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chat_user_id = gr.Text(
                        label="User ID", 
                        placeholder="Enter your user ID for personalized responses"
                    )
                with gr.Column(scale=1):
                    clear_chat_btn = gr.Button("Clear Chat", variant="secondary")
            
            chatbot = gr.Chatbot(
                label="Medical Assistant",
                height=500,
                type="messages",
                show_copy_button=True,  # Allow copying messages
                show_share_button=False,  # Disable sharing for privacy
                avatar_images=(
                    "https://cdn-icons-png.flaticon.com/512/147/147144.png",  # User avatar
                    "https://cdn-icons-png.flaticon.com/512/2040/2040946.png"  # Bot avatar (medical)
                ),
                bubble_full_width=False,  # Better message layout
                placeholder="<strong>Welcome to Medical AI Assistant!</strong><br>Ask me any medical questions. I have access to medical knowledge and your personal health data."
            )
            
            with gr.Row():
                with gr.Column(scale=4):
                    msg_box = gr.Textbox(
                        label="Your message", 
                        placeholder="Ask me anything about medical topics...",
                        lines=2,
                        max_lines=5,
                        autoscroll=False
                    )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send", variant="primary")

            
            send_btn.click(
                chat_bot,
                inputs=[msg_box, chatbot, chat_user_id],
                outputs=chatbot,
            ).then(
                clear_message,
                outputs=msg_box
            )
            
            msg_box.submit(
                chat_bot,
                inputs=[msg_box, chatbot, chat_user_id],
                outputs=chatbot,
            ).then(
                clear_message,
                outputs=msg_box
            )
            
            clear_chat_btn.click(
                clear_chat,
                outputs=chatbot
            )
            
            def handle_retry(history, retry_data: gr.RetryData, user_id: str):
                """Handle retry by regenerating the assistant's response"""
                try:
                    if not history or retry_data.index >= len(history):
                        return history
                    
                    user_message = None
                    for i in range(retry_data.index, -1, -1):
                        if history[i].get("role") == "user":
                            user_message = history[i]["content"]
                            break
                    
                    if not user_message:
                        return history
                    
                    new_history = history[:retry_data.index]
                    
                    return run_async(_chat_bot_async(user_message, new_history, user_id))
                    
                except Exception as e:
                    logger.error(f"Error in retry: {e}")
                    error_history = history.copy()
                    error_history.append({
                        "role": "assistant", 
                        "content": f"❌ Retry failed: {str(e)}"
                    })
                    return error_history

            chatbot.retry(
                handle_retry, 
                inputs=[chatbot, chat_user_id], 
                outputs=chatbot
            )
            
            def handle_undo(history, undo_data: gr.UndoData):
                """Handle undo by removing messages and restoring user input"""
                try:
                    if not history or undo_data.index >= len(history):
                        return history, ""
                    
                    user_message = history[undo_data.index]["content"] if undo_data.index < len(history) else ""
                    
                    new_history = history[:undo_data.index]
                    
                    return new_history, user_message
                    
                except Exception as e:
                    logger.error(f"Error in undo: {e}")
                    return history, ""

            chatbot.undo(
                handle_undo, 
                inputs=[chatbot], 
                outputs=[chatbot, msg_box]
            )
            
            def handle_like(like_data: gr.LikeData, user_id: str):
                """Handle like/dislike feedback"""
                try:
                    user_id = user_id.strip() or "anonymous"
                    feedback_type = "👍 liked" if like_data.liked else "👎 disliked"
                    message_content = like_data.value
                    
                    logger.info(f"User {user_id} {feedback_type} message: {message_content[:100]}...")
                    
                    feedback_data = {
                        "user_id": user_id,
                        "feedback": feedback_type,
                        "message": message_content,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }
                    
                    run_async(medical_kg.add_episode(
                        name=f"feedback-{user_id}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
                        episode_body=f"User feedback: {feedback_type} for message: {message_content}",
                        source_description=f"feedback:{user_id}",
                        source=EpisodeType.text,
                        reference_time=datetime.now(timezone.utc)
                    ))
                    
                    gr.Info(f"Thank you for your feedback! Message {feedback_type}.")
                    
                except Exception as e:
                    logger.error(f"Error handling like: {e}")
                    gr.Warning("Failed to record feedback.")

            chatbot.like(
                handle_like, 
                inputs=[chat_user_id], 
                outputs=None
            )
            
            def handle_edit(history, edit_data: gr.EditData, user_id: str):
                """Handle message editing"""
                try:
                    if not history or edit_data.index >= len(history):
                        return history
                    
                    new_history = history.copy()
                    new_history[edit_data.index]["content"] = edit_data.value
                    
                    if (edit_data.index < len(history) and 
                        history[edit_data.index].get("role") == "user"):
                        
                        new_history = new_history[:edit_data.index + 1]
                        
                        return run_async(_chat_bot_async(
                            edit_data.value, 
                            new_history[:-1], 
                            user_id
                        ))
                    
                    return new_history
                    
                except Exception as e:
                    logger.error(f"Error in edit: {e}")
                    return history

            chatbot.edit(
                handle_edit, 
                inputs=[chatbot, chat_user_id], 
                outputs=chatbot
            )

    return demo
