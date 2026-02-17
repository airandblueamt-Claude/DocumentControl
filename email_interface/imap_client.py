import logging
import time

from imapclient import IMAPClient

logger = logging.getLogger(__name__)

DEFAULT_RETRIES = 3
DEFAULT_RETRY_DELAY = 5


class ImapEmailClient:
    """IMAP client wrapper for mailbox operations with retry logic."""

    def __init__(self, auth_handler, host, port=993, retries=DEFAULT_RETRIES, retry_delay=DEFAULT_RETRY_DELAY):
        self.auth_handler = auth_handler
        self.host = host
        self.port = port
        self.retries = retries
        self.retry_delay = retry_delay
        self._client = None

    def connect(self):
        """Connect to the IMAP server and authenticate."""
        last_error = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.info("Connecting to %s:%s (attempt %d/%d)", self.host, self.port, attempt, self.retries)
                self._client = IMAPClient(self.host, port=self.port, ssl=True)
                self.auth_handler.login(self._client)
                logger.info("Connected successfully")
                return
            except Exception as e:
                last_error = e
                logger.warning("Connection attempt %d failed: %s", attempt, e)
                if attempt < self.retries:
                    time.sleep(self.retry_delay)

        raise ConnectionError(f"Failed to connect after {self.retries} attempts: {last_error}")

    def select_folder(self, folder='INBOX'):
        """Select a mailbox folder."""
        info = self._client.select_folder(folder)
        logger.info("Selected folder '%s' (%d messages)", folder, info.get(b'EXISTS', 0))
        return info

    def search_unread(self):
        """Search for unread messages, return message IDs."""
        msg_ids = self._client.search('UNSEEN')
        logger.info("Found %d unread messages", len(msg_ids))
        return msg_ids

    def search_all(self):
        """Search for ALL messages in the selected folder, return message IDs."""
        msg_ids = self._client.search('ALL')
        logger.info("Found %d total messages", len(msg_ids))
        return msg_ids

    def fetch_message(self, msg_id):
        """Fetch complete message in RFC822 format."""
        data = self._client.fetch([msg_id], ['RFC822'])
        return data[msg_id][b'RFC822']

    def mark_as_read(self, msg_id):
        """Mark a message as read (add \\Seen flag)."""
        self._client.add_flags([msg_id], [b'\\Seen'])

    def move_to_folder(self, msg_id, dest_folder):
        """Move a message to destination folder (COPY + DELETE + EXPUNGE)."""
        self.create_folder_if_missing(dest_folder)
        self._client.copy([msg_id], dest_folder)
        self._client.delete_messages([msg_id])
        self._client.expunge()
        logger.debug("Moved message %s to '%s'", msg_id, dest_folder)

    def create_folder_if_missing(self, folder_name):
        """Ensure a folder exists, create it if not."""
        if not self._client.folder_exists(folder_name):
            self._client.create_folder(folder_name)
            logger.info("Created folder '%s'", folder_name)

    def disconnect(self):
        """Gracefully close the IMAP connection."""
        if self._client:
            try:
                self._client.logout()
                logger.info("Disconnected from IMAP server")
            except Exception as e:
                logger.warning("Error during disconnect: %s", e)
            finally:
                self._client = None
