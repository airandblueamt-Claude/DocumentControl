import logging
import poplib
import time

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 5


class PopEmailClient:
    """POP3 client wrapper for mailbox operations with retry logic."""

    def __init__(self, username, password, host, port=995, retries=DEFAULT_RETRIES, retry_delay=DEFAULT_RETRY_DELAY):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = None

    def connect(self):
        """Connect to the POP3 server and authenticate."""
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.info("Connecting to %s:%s (attempt %d/%d)", self.host, self.port, attempt, self.retries)
                self._client = poplib.POP3_SSL(self.host, self.port)
                self._client.user(self.username)
                self._client.pass_(self.password)
                stat = self._client.stat()
                logger.info("Connected successfully - %d messages, %d bytes", stat[0], stat[1])
                return
            except Exception as e:
                last_error = e
                logger.warning("Connection attempt %d failed: %s", attempt, e)
                self._client = None
                if attempt < self.retries:
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Failed to connect after {self.retries} attempts: {last_error}")

    def list_messages(self):
        """List all messages, return list of (msg_number, size) tuples."""
        resp, listings, _ = self._client.list()
        messages = []
        for item in listings:
            line = item.decode() if isinstance(item, bytes) else item
            parts = line.split()
            msg_num = int(parts[0])
            msg_size = int(parts[1])
            messages.append((msg_num, msg_size))
        logger.info("Found %d messages in mailbox", len(messages))
        return messages

    def fetch_message(self, msg_num):
        """Fetch a complete message by number, return raw bytes."""
        resp, lines, octets = self._client.retr(msg_num)
        raw = b'\r\n'.join(lines)
        return raw

    def get_message_uid(self, msg_num):
        """Get unique ID for a message (survives across sessions)."""
        resp = self._client.uidl(msg_num)
        # Response format: b'+OK 1 uid_string'
        line = resp.decode() if isinstance(resp, bytes) else resp
        parts = line.split()
        if len(parts) >= 3:
            return parts[2]
        return parts[-1]

    def delete_message(self, msg_num):
        """Mark a message for deletion (applied on quit)."""
        self._client.dele(msg_num)
        logger.debug("Marked message %d for deletion", msg_num)

    def disconnect(self):
        """Gracefully close the POP3 connection (commits deletions)."""
        if self._client:
            try:
                self._client.quit()
                logger.info("Disconnected from POP3 server")
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
            finally:
                self._client = None
