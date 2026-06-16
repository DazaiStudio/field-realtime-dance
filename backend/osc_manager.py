from typing import Dict, List, Mapping, Optional

from pythonosc.udp_client import SimpleUDPClient

from osc_sender import OSCSender


class MultiSlotOSC:
    """Owns one OSCSender per slot (namespace /field/{slot}) plus a base
    client for meta messages (/field/active_slots, /field/count)."""

    def __init__(self, num_slots: int = 4, base_namespace: str = "/field",
                 host: str = "127.0.0.1", port: int = 9000, enabled: bool = True,
                 mode: str = "raw", alpha: float = 0.25):
        self.num_slots = num_slots
        self.base_namespace = base_namespace.rstrip("/") or "/field"
        self.host = host
        self.port = int(port)
        self.enabled = bool(enabled)
        self._senders: Dict[int, OSCSender] = {
            s: OSCSender(host=host, port=port, namespace=f"{self.base_namespace}/{s}",
                         enabled=enabled, mode=mode, alpha=alpha)
            for s in range(1, num_slots + 1)
        }
        self._meta_client = SimpleUDPClient(host, self.port)

    def sender(self, slot: int) -> OSCSender:
        return self._senders[slot]

    def send_slot(self, slot: int, metrics: Mapping[str, float]) -> None:
        self._senders[slot].send_metrics(metrics)

    def send_named_slot(self, slot: int, name: str, value: float) -> None:
        self._senders[slot].send_named(name, value)

    def send_meta(self, active_slots: List[int]) -> None:
        if not self.enabled:
            return
        try:
            self._meta_client.send_message(f"{self.base_namespace}/active_slots", active_slots)
            self._meta_client.send_message(f"{self.base_namespace}/count", len(active_slots))
        except Exception as exc:
            print(f"OSC meta send failed: {exc}")

    def configure(self, host: Optional[str] = None, port: Optional[int] = None,
                  enabled: Optional[bool] = None, mode: Optional[str] = None,
                  alpha: Optional[float] = None) -> None:
        if host is not None:
            self.host = host
        if port is not None:
            self.port = int(port)
        if enabled is not None:
            self.enabled = bool(enabled)
        for s in self._senders.values():
            s.configure(host=host, port=port, enabled=enabled, mode=mode, alpha=alpha)
        if host is not None or port is not None:
            self._meta_client = SimpleUDPClient(self.host, self.port)

    def prepared_for(self, slot: int) -> Dict[str, float]:
        return dict(self._senders[slot].last_prepared_metrics)
