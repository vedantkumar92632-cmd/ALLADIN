import time
from core.logger import get_logger
from config.settings import settings
from engine.clock import clock
from engine.cache import cache
from engine.message_bus import message_bus
from engine.event_types import SystemStartedEvent, SystemStoppedEvent

logger = get_logger("engine.kernel")

class Kernel:
    def __init__(self):
        self.is_running = False

    def start(self):
        """Boot the system: wire logger, clock, cache, message bus together."""
        if self.is_running:
            logger.warning("Kernel is already running.")
            return

        logger.info("Booting ARTHA V1 Kernel...")
        
        # 1. Initialize clock mode from settings
        clock_mode = settings.get("clock.mode", "LIVE")
        clock.set_mode(clock_mode)
        logger.info(f"System clock initialized in {clock.get_mode()} mode.")

        # 2. Verify cache is responsive
        cache.set("system_heartbeat", time.time())
        logger.info("Cache verified.")

        # 3. Log data domain deferrals
        logger.info("Ingestion Deferral Notice: Domain 3 (Fundamentals/Filings) and Domain 4 (News/NLP text feeds) are deferred under the current build scope (V1 EOD focus).")

        self.is_running = True
        logger.info("ARTHA V1 Kernel booted successfully.")

        # 4. Publish SystemStartedEvent
        env = settings.get("general.env", "development")
        message_bus.publish(SystemStartedEvent(env=env))

    def stop(self, reason: str = "normal"):
        """Shut down the system cleanly."""
        if not self.is_running:
            logger.warning("Kernel is not running.")
            return

        logger.info(f"Shutting down ARTHA V1 Kernel... Reason: {reason}")
        
        # Publish SystemStoppedEvent before shutting down
        message_bus.publish(SystemStoppedEvent(reason=reason))

        self.is_running = False
        logger.info("ARTHA V1 Kernel shut down cleanly.")

# Global kernel instance
kernel = Kernel()
