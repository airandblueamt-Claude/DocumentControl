import logging
import os
import atexit

import msal

logger = logging.getLogger(__name__)


class AuthHandler:
    """Base class - all auth handlers implement this interface."""

    def login(self, imap_connection):
        """Authenticate the IMAP connection. Raises on failure."""
        raise NotImplementedError


class BasicAuthHandler(AuthHandler):
    """Simple username + password login (for non-O365 servers)."""

    def __init__(self, username, password):
        self.username = username
        self.password = password

    def login(self, imap_connection):
        imap_connection.login(self.username, self.password)
        logger.info("Authenticated via basic auth as %s", self.username)


class O365OAuthHandler(AuthHandler):
    """OAuth2 via MSAL device code flow (required for Office 365)."""

    def __init__(self, tenant_id, client_id, scopes, email, token_cache_path):
        self.email = email
        self.scopes = scopes
        self.token_cache_path = token_cache_path

        self._cache = msal.SerializableTokenCache()
        if os.path.exists(token_cache_path):
            with open(token_cache_path, 'r') as f:
                self._cache.deserialize(f.read())

        authority = f"https://login.microsoftonline.com/{tenant_id}"
        self._app = msal.PublicClientApplication(
            client_id,
            authority=authority,
            token_cache=self._cache,
        )

        atexit.register(self._save_cache)

    def login(self, imap_connection):
        """Authenticate IMAP connection using OAuth2 bearer token."""
        token = self._get_access_token()
        imap_connection.oauth2_login(self.email, token)
        logger.info("Authenticated via OAuth2 as %s", self.email)

    def _get_access_token(self):
        """Get a valid access token (silent refresh or device code flow)."""
        accounts = self._app.get_accounts()
        if accounts:
            result = self._app.acquire_token_silent(self.scopes, account=accounts[0])
            if result and 'access_token' in result:
                logger.debug("Token acquired silently from cache")
                return result['access_token']

        result = self._authenticate_device_flow()
        if 'access_token' in result:
            self._save_cache()
            return result['access_token']

        error = result.get('error_description', result.get('error', 'Unknown error'))
        raise RuntimeError(f"OAuth2 authentication failed: {error}")

    def _authenticate_device_flow(self):
        """First-time auth: user visits URL, enters code."""
        flow = self._app.initiate_device_flow(scopes=self.scopes)
        if 'user_code' not in flow:
            raise RuntimeError(f"Device flow initiation failed: {flow.get('error_description', 'Unknown error')}")

        print("\n" + "=" * 60)
        print("AUTHENTICATION REQUIRED")
        print("=" * 60)
        print(flow['message'])
        print("=" * 60 + "\n")

        result = self._app.acquire_token_by_device_flow(flow)
        return result

    def _save_cache(self):
        """Persist token cache to disk."""
        if self._cache.has_state_changed:
            os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
            with open(self.token_cache_path, 'w') as f:
                f.write(self._cache.serialize())
            logger.debug("Token cache saved to %s", self.token_cache_path)


def create_auth_handler(auth_config, email_address):
    """Factory: reads auth_method from config, returns the right handler."""
    method = auth_config['auth_method']

    if method == 'basic':
        return BasicAuthHandler(auth_config['username'], auth_config['password'])
    elif method == 'oauth2':
        return O365OAuthHandler(
            tenant_id=auth_config['tenant_id'],
            client_id=auth_config['client_id'],
            scopes=auth_config['scopes'],
            email=email_address,
            token_cache_path=auth_config.get('token_cache_path', 'config/.token_cache.bin'),
        )
    else:
        raise ValueError(f"Unknown auth_method: {method}")
