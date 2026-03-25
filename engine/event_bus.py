from collections import defaultdict
from typing import Callable, Dict, List

from .event import Event, EventType


class EventBus:
	def __init__(self):
		self._listeners: Dict[EventType, List[Callable]] = defaultdict(list)
		self._priority_listeners: Dict[EventType, List[Callable]] = defaultdict(list)

	def subscribe(
			self, event_type: EventType, handler: Callable, priority: bool = False
	):
		if priority:
			self._priority_listeners[event_type].append(handler)
		else:
			self._listeners[event_type].append(handler)

	def unsubscribe(self, event_type: EventType, handler: Callable):
		if handler in self._listeners[event_type]:
			self._listeners[event_type].remove(handler)
		if handler in self._priority_listeners[event_type]:
			self._priority_listeners[event_type].remove(handler)

	def emit(self, event: Event) -> Event:
		handlers = self._priority_listeners[event.type] + self._listeners[event.type]

		for handler in handlers:
			if event.is_cancelled():
				break
			result = handler(event)
			if result is not None and isinstance(result, Event):
				event = result

		return event

	def clear(self):
		self._listeners.clear()
		self._priority_listeners.clear()

	def has_listeners(self, event_type: EventType) -> bool:
		return (
				len(self._listeners[event_type]) > 0
				or len(self._priority_listeners[event_type]) > 0
		)
