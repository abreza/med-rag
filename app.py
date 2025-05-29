import logging
import signal
import sys
from pathlib import Path

from config import config
from knowledge_graph import medical_kg
from utils.logging import setup_logging

async def initialize_services():
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Initializing Medical Knowledge Graph...")
        await medical_kg.initialize()
        
        if await medical_kg.health_check():
            logger.info("✅ Medical Knowledge Graph is healthy")
        else:
            logger.warning("⚠️ Medical Knowledge Graph health check failed")
            
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}")
        raise


async def cleanup_services():
    logger = logging.getLogger(__name__)
    setup_logging()
    
    try:
        logger.info("Shutting down services...")
        await medical_kg.close()
        logger.info("✅ Services shut down successfully")
    except Exception as e:
        logger.error(f"❌ Error during cleanup: {e}")


def setup_signal_handlers(async_manager):
    logger = logging.getLogger(__name__)
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        try:
            async_manager.run_async(cleanup_services())
        except Exception as e:
            logger.error(f"Error during signal cleanup: {e}")
        finally:
            async_manager.shutdown()
            sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    logger = logging.getLogger(__name__)
    
    from ui import create_interface, async_manager
    setup_signal_handlers(async_manager)
    async_manager.run_async(initialize_services())
    app = create_interface()
    app.launch(
        server_name=config.app.host,
        server_port=config.app.port,
        share=config.app.share,
        debug=config.app.debug,
        show_error=config.app.debug,
        quiet=not config.app.debug
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)
