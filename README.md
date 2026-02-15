# Document Control Email Interface

Automated email processing tool for document control. Connects to a mailbox via IMAP, classifies incoming emails, saves attachments in an organized folder structure, and logs everything to an Excel spreadsheet with transmittal numbers.

## Features

- Connects to Office 365 (or any IMAP server) to fetch unread emails
- Classifies emails by document type and discipline using keyword matching
- Extracts reference numbers from subjects and body text
- Saves attachments to organized folders: `YYYY/MM-Month/TRN-YYYY-NNNN/`
- Generates sequential transmittal numbers (TRN-2026-0001, TRN-2026-0002, ...)
- Logs all correspondence to an Excel file (`logs/correspondence_log.xlsx`)
- Tracks processed emails in SQLite to prevent duplicates
- Skips auto-replies, out-of-office, and delivery notifications
- Detects emails requiring a response (urgent, action required, etc.)

## Project Structure

```
DocumentControl/
├── email_monitor.py                  # Main entry point
├── requirements.txt                  # Python dependencies
├── config/
│   ├── email_config.yaml             # Server, auth, polling, attachment settings
│   └── classification_config.yaml    # Classification rules, skip filters
├── email_interface/
│   ├── __init__.py
│   ├── auth.py                       # Authentication (OAuth2 + basic auth)
│   ├── imap_client.py                # IMAP connection with retry logic
│   ├── message_processor.py          # Email parsing & classification
│   ├── attachment_handler.py         # Attachment saving & folder organization
│   ├── persistence.py                # SQLite tracking & transmittal numbers
│   └── config.py                     # YAML config loader
├── logs/                             # Log files & Excel output (auto-created)
├── data/                             # SQLite database (auto-created)
└── Attachments/                      # Saved attachments (auto-created)
```

## Setup on Linux

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

```bash
python3 --version    # Verify Python is installed
```

### 2. Clone / Copy the Project

```bash
cd ~
# If using git:
git clone <repo-url> DocumentControl
# Or copy the folder to your home directory
```

### 3. Create a Virtual Environment

```bash
cd ~/DocumentControl
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `imapclient` - IMAP protocol handling
- `msal` - Microsoft OAuth2 authentication
- `openpyxl` - Excel file reading/writing
- `pyyaml` - YAML config parsing
- `python-dateutil` - Date parsing

### 5. Configure Your Email Account

Edit `config/email_config.yaml`:

**For basic authentication (username + password):**

```yaml
auth:
  auth_method: basic
  username: "your-email@example.com"
  password: "your-password"

mailbox:
  email_address: "your-email@example.com"
```

**For OAuth2 (Office 365 with modern auth):**

```yaml
auth:
  auth_method: oauth2
  tenant_id: "your-azure-tenant-id"
  client_id: "your-azure-client-id"
  scopes:
    - "https://outlook.office365.com/IMAP.AccessAsUser.All"
    - "offline_access"
  token_cache_path: "config/.token_cache.bin"

mailbox:
  email_address: "your-email@example.com"
```

> **Note:** Microsoft has disabled basic auth for most Office 365 tenants. If basic auth fails with an authentication error, you will need to use OAuth2 with an Azure AD app registration.

### 6. Configure Classification Rules (Optional)

Edit `config/classification_config.yaml` to customize:

- `type_keywords` - Document type classification (RFI, Submittal, Civil, etc.)
- `discipline_keywords` - Discipline tagging (Electrical, Networking, etc.)
- `ref_regexes` - Reference number extraction patterns
- `skip_filters` - Emails to automatically skip (auto-replies, noreply senders)
- `response_required_phrases` - Phrases that flag emails needing a response

## Running

### Single Run

```bash
cd ~/DocumentControl
source venv/bin/activate
python3 email_monitor.py
```

### Scheduled Execution with Cron

To run every 10 minutes:

```bash
crontab -e
```

Add this line:

```
*/10 * * * * cd /home/malkhalifa/DocumentControl && /home/malkhalifa/DocumentControl/venv/bin/python3 email_monitor.py >> /home/malkhalifa/DocumentControl/logs/cron.log 2>&1
```

### Running as a Background Service (systemd)

Create `/etc/systemd/system/doccontrol-email.service`:

```ini
[Unit]
Description=Document Control Email Monitor
After=network.target

[Service]
Type=oneshot
User=malkhalifa
WorkingDirectory=/home/malkhalifa/DocumentControl
ExecStart=/home/malkhalifa/DocumentControl/venv/bin/python3 email_monitor.py

[Install]
WantedBy=multi-user.target
```

Create a timer `/etc/systemd/system/doccontrol-email.timer`:

```ini
[Unit]
Description=Run Document Control Email Monitor every 10 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
```

Enable it:

```bash
sudo systemctl enable --now doccontrol-email.timer
```

## Output

- **Excel log:** `logs/correspondence_log.xlsx` - All processed emails with transmittal numbers, sender, recipients, type, discipline, etc.
- **Attachments:** `Attachments/YYYY/MM-Month/TRN-YYYY-NNNN/` - Organized by date and transmittal number
- **Application log:** `logs/email_interface.log` - Detailed processing log with timestamps
- **SQLite database:** `data/tracking.db` - Tracks processed message IDs and transmittal number sequences

## Switching Between Auth Methods

In `config/email_config.yaml`, change `auth_method`:

```yaml
# To use basic auth:
auth:
  auth_method: basic
  username: "your-email@example.com"
  password: "your-password"

# To use OAuth2:
auth:
  auth_method: oauth2
  tenant_id: "..."
  client_id: "..."
```

Comment out the settings you're not using to keep the config clean.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `LOGIN failed` | Check username/password. For O365, basic auth may be disabled — switch to OAuth2 |
| `Connection refused` | Verify `imap_host` and `imap_port` (993 for SSL) |
| `ModuleNotFoundError` | Activate the virtual environment: `source venv/bin/activate` |
| `FileNotFoundError: config/...` | Run from the project root: `cd ~/DocumentControl` |
| OAuth2 device flow hangs | Open the URL shown in terminal, enter the code, then wait |
