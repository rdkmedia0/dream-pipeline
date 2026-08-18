"""
Local-machine encryption for secrets this pipeline stores at rest --
currently just YouTube's client_secret.json/token.json (upload_dream.py),
but written generically so any future secret can reuse it.

HOW IT WORKS
------------
A single symmetric key (Fernet -- AES128-CBC + HMAC, from the standard
`cryptography` package, already an installed dependency via google-auth)
is generated once and stored OUTSIDE this project folder entirely, in
the same kind of per-user local-app-data directory Comfy Desktop uses
for its own state (see dream_step._comfy_desktop_appdata_dir() for the
same cross-platform pattern) -- specifically so the key can never end up
inside a git-tracked directory or get swept up in a project-folder copy/
backup by accident. Encrypted secrets themselves (client_secret.json.enc,
token.json.enc) DO live inside the normal project/pipeline folders next
to everything else, but are useless without this separately-stored key.

THREAT MODEL -- BE HONEST ABOUT WHAT THIS DOES AND DOESN'T PROTECT
--------------------------------------------------------------------
This protects secrets from being readable by casually opening the JSON
file (e.g. a screen-share, a careless `cat`, an accidental git add, a
cloud-synced project folder). It does NOT protect against a real local
attacker with access to this same Windows user account -- they can read
the key file exactly as this code does. Full protection would need an
OS-level secret store (Windows Credential Manager / macOS Keychain);
this is a deliberately lighter-weight "not plaintext on disk" measure
consistent with this being a local single-user tool, not a multi-tenant
service. Documented plainly in help.html -- never oversold as more than
this actually is.
"""
import os
import platform
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


def _local_appdata_dir():
    """This pipeline's own per-user local state directory -- same
    cross-platform convention as dream_step._comfy_desktop_appdata_dir(),
    but for this app's own secrets, never Comfy's.

    DREAM_PIPELINE_CONFIG_DIR (same env var dream_step.CONFIG_PATH
    already respects) overrides this entirely when set -- added for the
    Docker image, where $HOME is ephemeral container filesystem, not a
    mounted volume: without this, the Fernet key would regenerate on
    every container recreation, permanently orphaning any .enc files
    that were encrypted with the previous key (exactly the
    present-but-undecryptable failure mode this project already hit
    once, during the Windows->Linux port -- a container restart would
    silently reproduce it every time otherwise)."""
    override = os.environ.get("DREAM_PIPELINE_CONFIG_DIR")
    if override:
        return Path(override)
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return base / "Dream Pipeline"


_KEY_PATH = _local_appdata_dir() / "secret.key"


def _get_or_create_key():
    """The Fernet key, generating one on first use. File permissions are
    tightened to owner-read/write-only where the OS supports it (POSIX
    chmod 600) -- a no-op on Windows, where NTFS ACLs already default to
    the owning user's profile folder not being readable by other local
    accounts."""
    if _KEY_PATH.is_file():
        return _KEY_PATH.read_bytes()
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    _KEY_PATH.write_bytes(key)
    try:
        os.chmod(_KEY_PATH, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # best-effort on platforms/filesystems that don't support it
    return key


def encrypt_text(text):
    """Returns encrypted bytes -- write these directly to a `.enc` file."""
    return Fernet(_get_or_create_key()).encrypt(text.encode("utf-8"))


def write_encrypted(path, text):
    """The ONLY way a `.enc` secret file should ever be written -- encrypts
    and chmods 0600 (POSIX, best-effort, same no-op-on-Windows reasoning as
    _get_or_create_key's own chmod) in one place. Confirmed real gap
    (2026-08-18): every call site used to do
    `path.write_bytes(secret_store.encrypt_text(text))` directly, which
    writes the encrypted payload with the OS/filesystem's default
    permissions -- only the Fernet KEY file itself ever got tightened to
    owner-only, leaving every actual secret payload (client_secret.json.enc,
    token.json.enc, gemini_api_key.enc) relying purely on directory-level
    ACLs. Centralizing the write here means a future secret can't be added
    without this protection by accident."""
    path = Path(path)
    path.write_bytes(encrypt_text(text))
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass  # best-effort, same as _get_or_create_key's own chmod


def decrypt_text(data):
    """Reverses encrypt_text(). Raises ValueError (not Fernet's own
    InvalidToken) on a bad/corrupt/wrong-key file, so callers can catch
    one clear exception type regardless of the underlying library."""
    try:
        return Fernet(_get_or_create_key()).decrypt(data).decode("utf-8")
    except InvalidToken:
        raise ValueError(
            "Could not decrypt -- the local encryption key "
            f"({_KEY_PATH}) doesn't match what this file was encrypted "
            "with (a different machine, a reinstalled/regenerated key, "
            "or a corrupted file). The secret needs to be re-added.")


def decrypt_status(path):
    """(present, decryptable, reason) for an .enc file -- a cheap, local,
    no-network check proving the file can ACTUALLY be used, not just
    that it exists. Every *_key_status/*_token_status endpoint in
    web_ui.py used to report "present": path.is_file() alone, which is
    a false-green: confirmed live 2026-08-15, post Windows->Linux port,
    that a Gemini key .enc file survives copying the project folder over
    but the Fernet key at _KEY_PATH does NOT (it deliberately lives
    OUTSIDE the project folder, precisely so it's never swept up in a
    copy/backup -- see this file's own module docstring), so the ported
    copy's .enc is present on disk yet permanently undecryptable there.
    The GUI showed a green "ENABLED" badge for a key that failed on the
    very first real use. present=False when the file doesn't exist at
    all (a different, unrelated case -- "never configured", not "broken
    copy"); decryptable/reason are None in that case."""
    p = Path(path)
    if not p.is_file():
        return False, None, None
    try:
        decrypt_text(p.read_bytes())
        return True, True, None
    except Exception as e:
        return True, False, str(e)


def migrate_plaintext_if_present(plaintext_path, encrypted_path):
    """One-time upgrade path: if an old plaintext secret file exists
    (from before this module existed) and the new encrypted path
    doesn't, encrypts it in place and removes the plaintext original.
    Safe to call on every startup -- a no-op once migrated. Never
    silently overwrites an existing .enc file (if both somehow exist,
    the .enc file wins and the plaintext is left alone rather than
    guessing which is newer)."""
    plaintext_path = Path(plaintext_path)
    encrypted_path = Path(encrypted_path)
    if encrypted_path.is_file() or not plaintext_path.is_file():
        return
    write_encrypted(encrypted_path, plaintext_path.read_text(encoding="utf-8"))
    plaintext_path.unlink()
