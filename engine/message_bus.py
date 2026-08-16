from collections import defaultdict
from typing import Callable, Dict, List, Type, Any
from engine.event_types import Event
from core.logger import get_logger

logger = get_logger("engine.message_bus")

class MessageBus:
    def __init__(self):
        # Maps an event class type to a list of handler callbacks
        self._subscribers: Dict[Type[Event], List[Callable[[Any], None]]] = defaultdict(list)

    def subscribe(self, event_type: Type[Event], handler: Callable[[Any], None]):
        """Subscribe a handler callback to a specific event type."""
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
            logger.debug(f"Subscribed handler to event {event_type.__name__}")

    def unsubscribe(self, event_type: Type[Event], handler: Callable[[Any], None]):
        """Unsubscribe a handler from a specific event type."""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            logger.debug(f"Unsubscribed handler from event {event_type.__name__}")

    def publish(self, event: Event):
        """Publish an event to all subscribed handlers of its type."""
        event_type = type(event)
        handlers = self._subscribers.get(event_type, [])
        
        # Also match subscribers to base Event class if any
        if event_type is not Event:
            handlers = list(handlers) + list(self._subscribers.get(Event, []))
            handlers = list(dict.fromkeys(handlers))
            
        if not handlers:
            logger.debug(f"No subscribers registered for event {event_type.__name__}")
            return
            
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Error handling event {event_type.__name__}: {e}", exc_info=True)

# Global message bus instance
message_bus = MessageBus()
