"""
Claude Code messaging protocol: session discovery (~/.claude/sessions),
Unix-domain-socket transport (auth + message framing, ack), and the
send/receive primitives used to talk to other Claude Code / Antigravity
sessions.

This module has no knowledge of MCP, stdio, or the CLI - it is the
reusable core that both the MCP entry point (mcp_server.py) and the
standalone daemon entry point (standalone.py) build on top of.
"""
import os
import re
import json
import glob
import uuid
import stat
import time
import hashlib
import asyncio
import logging
import psutil
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("SocketBridgeMCP")


class ClaudeMessagingProtocol:
    def __init__(
        self,
        bridge_socket_path: Optional[str] = None,
        session_name: Optional[str] = None,
        on_inbound_message: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        """
        on_inbound_message(method, params), if given, is invoked whenever a real
        inbound message frame is processed - the hook a caller uses to forward
        activity onward (e.g. as an MCP notification). Purely optional: this
        class has no idea what, if anything, is listening on the other end.
        """
        self.pid = os.getpid()
        self.home_dir = os.path.expanduser("~")
        self.sessions_dir = os.path.join(self.home_dir, ".claude", "sessions")
        self.on_inbound_message = on_inbound_message

        env_name = os.environ.get("AGY_SESSION_NAME")
        self.session_name = session_name or env_name or self._derive_default_session_name()

        env_socket = os.environ.get("AGY_MCP_BRIDGE_SOCKET")
        if bridge_socket_path:
            self.bridge_socket_path = bridge_socket_path
        elif env_socket:
            self.bridge_socket_path = env_socket
        else:
            self.bridge_socket_path = self._get_default_socket_path(self.pid, self.session_name)

        # Historical datastore for storing response payloads
        self.response_store: Dict[str, Dict[str, Any]] = {}
        self.active_peers: Dict[str, Dict[str, Any]] = {}

    def _derive_default_session_name(self) -> str:
        """
        Derives a stable, per-workspace default session name from cwd so that
        unrelated bridge instances (different Antigravity workspaces) never
        collide on the same socket, while the same workspace reconnecting
        across MCP host restarts always resolves back to the same identity.
        """
        cwd_digest = hashlib.sha1(os.getcwd().encode("utf-8")).hexdigest()[:8]
        return f"antigravity-bridge-{cwd_digest}"

    def _get_default_socket_path(self, pid: int, session_name: Optional[str] = None) -> str:
        sname = (session_name or getattr(self, "session_name", "antigravity-bridge")).replace("/", "-")
        try:
            uid = os.getuid()
            cc_socks_dir = f"/run/user/{uid}/cc-socks"
            if os.path.exists(cc_socks_dir) and os.access(cc_socks_dir, os.W_OK):
                return os.path.join(cc_socks_dir, f"bridge-{sname}.sock")

            fallback_dir = "/tmp/cc-socks"
            os.makedirs(fallback_dir, exist_ok=True)
            return os.path.join(fallback_dir, f"bridge-{sname}.sock")
        except Exception:
            pass
        return f"/tmp/bridge_{sname}.sock"

    def _is_pid_alive(self, pid: Optional[int]) -> bool:
        if not pid or not isinstance(pid, int):
            return False
        try:
            # A zombie still satisfies psutil.pid_exists()/os.kill(pid, 0) - it occupies
            # a process table entry until its parent reaps it - so either check alone
            # can report a daemon we just killed as still "alive" indefinitely. Check
            # its actual status too.
            proc = psutil.Process(pid)
            return proc.status() != psutil.STATUS_ZOMBIE
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
        except Exception:
            return False

    async def verify_and_purge_sessions(self) -> Dict[str, Dict[str, Any]]:
        """
        Scans workspace configurations, checks if tracking PIDs are alive,
        and systematically cleans up orphaned or dead sockets and key files.
        """
        logger.info("Running proactive session structural analysis...")
        if not os.path.exists(self.sessions_dir):
            logger.warning(f"Sessions directory does not exist: {self.sessions_dir}")
            self.active_peers = {}
            return {}

        current_sessions: Dict[str, Dict[str, Any]] = {}
        json_files = glob.glob(os.path.join(self.sessions_dir, "*.json"))
        purged_count = 0

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                pid = meta.get("pid")
                sock_path = meta.get("messagingSocketPath")
                name = meta.get("name", f"session-{pid}")
                session_id = meta.get("sessionId", "")
                cwd = meta.get("cwd", "")

                # Check process lifecycle
                if pid and self._is_pid_alive(pid):
                    # Check for accompanying auth key file
                    key_pattern = os.path.join(self.sessions_dir, f"{pid}.*.key")
                    discovered_keys = glob.glob(key_pattern)

                    token = ""
                    if discovered_keys:
                        try:
                            with open(discovered_keys[0], "r", encoding="utf-8") as kf:
                                key_raw = kf.read().strip()
                                try:
                                    key_data = json.loads(key_raw)
                                    if isinstance(key_data, dict) and "peerToken" in key_data:
                                        token = key_data["peerToken"]
                                    else:
                                        token = key_raw
                                except json.JSONDecodeError:
                                    token = key_raw
                        except Exception as kerr:
                            logger.error(f"Failed to read key file {discovered_keys[0]}: {kerr}")

                    current_sessions[name] = {
                        "name": name,
                        "pid": pid,
                        "sessionId": session_id,
                        "socket": sock_path,
                        "token": token,
                        "cwd": cwd,
                        "kind": meta.get("kind", "interactive"),
                        "updatedAt": meta.get("updatedAt"),
                        "status": meta.get("status", "active")
                    }
                else:
                    # Target process is dead; purge stale descriptor and key files safely
                    logger.warning(f"Purging dead session files for PID {pid}")
                    try:
                        os.remove(file_path)
                        purged_count += 1
                    except OSError as e:
                        logger.error(f"Failed to remove stale session file {file_path}: {e}")

                    # Also purge associated key files for dead PID
                    if pid:
                        key_pattern = os.path.join(self.sessions_dir, f"{pid}.*.key")
                        for key_file in glob.glob(key_pattern):
                            try:
                                os.remove(key_file)
                                logger.info(f"Purged stale key file: {key_file}")
                            except OSError as e:
                                logger.error(f"Failed to remove stale key file {key_file}: {e}")
            except Exception as e:
                logger.error(f"Error checking session file {file_path}: {e}")

        self.active_peers = current_sessions
        logger.info(f"Session analysis complete. Active peers: {len(current_sessions)}, Purged: {purged_count}")
        return self.active_peers

    def _resolve_session(self, session_identifier: str) -> Optional[Dict[str, Any]]:
        if not self.active_peers:
            return None

        # Direct match by name
        if session_identifier in self.active_peers:
            return self.active_peers[session_identifier]

        # Match by PID (str or int)
        for peer in self.active_peers.values():
            if str(peer.get("pid")) == str(session_identifier):
                return peer

        # Match by SessionID
        for peer in self.active_peers.values():
            if peer.get("sessionId") == session_identifier:
                return peer

        # Substring / case-insensitive match on name
        session_lower = session_identifier.lower()
        for name, peer in self.active_peers.items():
            if session_lower in name.lower():
                return peer

        # If user passed "default", "first", or "active", return first active session
        if session_identifier.lower() in ["default", "first", "active", "any"]:
            return next(iter(self.active_peers.values()))

        return None

    async def send_to_claude(
        self,
        session_identifier: str,
        message_content: str
    ) -> Dict[str, Any]:
        """
        Establishes an out-bound wire pipe to a verified target session and dispatches
        a message. Fire-and-forget: if the target sends a real reply, it arrives later
        as its own inbound frame (see handle_inbound_client) - there is no in_reply_to
        correlation in the wire protocol, so this does not (and cannot honestly) wait
        for or return that reply synchronously.
        """
        await self.verify_and_purge_sessions()

        peer = self._resolve_session(session_identifier)
        if not peer:
            logger.error(f"Target session '{session_identifier}' is not reachable.")
            return {
                "success": False,
                "error": f"Target session '{session_identifier}' not found among active sessions: {list(self.active_peers.keys())}"
            }

        socket_path = peer.get("socket")

        if not socket_path or not os.path.exists(socket_path):
            logger.error(f"Socket path for '{peer['name']}' does not exist: {socket_path}")
            return {
                "success": False,
                "error": f"Socket path '{socket_path}' for session '{peer['name']}' does not exist on disk."
            }

        # Wire format reverse-engineered from a real Claude Code SendMessage
        # frame (captured directly off the socket): no separate auth frame,
        # a "from" field holding this sender's own "uds:<socket path>" URI
        # (not a bare name), a msgV version field, and the message content
        # itself wrapped in a <cross-session-message from=... from-name=...
        # from-mode=...> tag - the receiving side's pretty, reply-able
        # rendering keys off that wrapper being present in the content, not
        # off any separate top-level metadata field. A prior version omitted
        # all of this and sent a bare "sender" name instead of "from" -
        # real Claude Code sessions still buffered that content correctly,
        # but rendered it as plain unwrapped text with no reply-to address,
        # and (unconfirmed but consistent with every observed case) reported
        # send failure back to the sender despite the content having arrived.
        msg_id = str(uuid.uuid4())
        from_uri = f"uds:{self.bridge_socket_path}"
        wrapped_content = (
            f'<cross-session-message from="{from_uri}" from-name="{self.session_name}" '
            f'from-mode="prompting">\n{message_content}\n</cross-session-message>'
        )
        message_frame = {
            "msgV": 1,
            "msg_id": msg_id,
            "type": "user",
            "message": {"role": "user", "content": wrapped_content},
            "priority": "next",
            "from": from_uri
        }

        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write((json.dumps(message_frame) + "\n").encode("utf-8"))
            await writer.drain()

            try:
                ack_line = await asyncio.wait_for(reader.readline(), timeout=2.0)
                if ack_line:
                    logger.info(f"Received instant ACK from {peer['name']}: {ack_line.decode('utf-8').strip()}")
            except (asyncio.TimeoutError, Exception):
                pass

            writer.close()
            await writer.wait_closed()

            entry = {
                "direction": "outbound",
                "status": "dispatched",
                "msg_id": msg_id,
                "target_session": peer["name"],
                "target_pid": peer["pid"],
                "content": message_content,
                "response": ""
            }
            self.response_store[msg_id] = entry
            logger.info(f"Successfully dispatched message {msg_id} to {peer['name']}")

            return {
                "success": True,
                "msg_id": msg_id,
                "target": peer["name"],
                "status": "dispatched",
                "detail": entry
            }

        except Exception as e:
            logger.error(f"IPC injection crash sending to {socket_path}: {e}")
            return {
                "success": False,
                "error": f"Failed to send message over Unix socket to {socket_path}: {e}"
            }

    def _compute_pid_domain(self, pid: int) -> str:
        """
        Builds the pidDomain field the way real Claude Code sessions do:
        "linux:<contents of /etc/machine-id>:pid:[<this pid's PID-namespace
        inode, from /proc/<pid>/ns/pid>]" - verified by comparing a real
        session's descriptor against this host's actual /etc/machine-id and
        /proc/<pid>/ns/pid. A prior version fabricated "linux:local:pid:[pid]"
        (the literal PID, not its namespace inode, and a placeholder instead
        of the real machine id) - plausibly why native cross-session
        SendMessage could resolve this bridge by name but fail to actually
        deliver to it. Falls back to that placeholder shape only if either
        file is unreadable (e.g. non-Linux).
        """
        try:
            with open("/etc/machine-id", "r", encoding="utf-8") as f:
                machine_id = f.read().strip()
        except OSError:
            machine_id = "local"
        try:
            ns_link = os.readlink(f"/proc/{pid}/ns/pid")  # "pid:[4026532221]"
            ns_part = ns_link.split(":", 1)[1] if ":" in ns_link else f"[{pid}]"
        except OSError:
            ns_part = f"[{pid}]"
        return f"linux:{machine_id}:pid:{ns_part}"

    def _compute_proc_start(self, pid: int) -> str:
        """
        Real Claude Code descriptors put /proc/<pid>/stat's 22nd field (process
        start time in clock ticks since boot - the kernel's own anti-PID-reuse
        value) in procStart, not the pid itself - verified against a real
        session's descriptor (its procStart matched that field exactly; the
        pid did not). Falls back to str(pid) only if /proc is unreadable.
        """
        try:
            with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as f:
                # Field 2 (comm) is parenthesized and may itself contain
                # spaces/parens, so split after its closing paren rather than
                # by naive whitespace splitting from the start.
                after_comm = f.read().rsplit(")", 1)[1].split()
                return after_comm[19]  # fields are 1-indexed; index from field 3 onward
        except (OSError, IndexError):
            return str(pid)

    def register_session_descriptor(self, session_name: str = "antigravity-bridge", kind: str = "bg"):
        """
        Registers a session descriptor in ~/.claude/sessions/ so that surrounding
        Claude Code processes can discover and message this bridge via ListAgents.
        """
        if not os.path.exists(self.sessions_dir):
            try:
                os.makedirs(self.sessions_dir, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create sessions directory: {e}")
                return

        self.registered_pid = os.getpid()
        self.registered_session_id = str(uuid.uuid4())
        self.registered_token = uuid.uuid4().hex
        now = int(time.time() * 1000)

        sanitized_name = session_name.replace("/", "-")
        self.session_json_path = os.path.join(self.sessions_dir, f"{self.registered_pid}.json")
        self.session_key_path = os.path.join(self.sessions_dir, f"{self.registered_pid}.{self.registered_token[:16]}.key")
        self.named_descriptor_path = os.path.join(self.sessions_dir, f"bridge-{sanitized_name}.json")

        session_data = {
            "pid": self.registered_pid,
            "sessionId": self.registered_session_id,
            "cwd": os.getcwd(),
            "startedAt": now,
            "procStart": self._compute_proc_start(self.registered_pid),
            "version": "2.1.282",
            "peerProtocol": 1,
            "peerFeatures": ["notify_idle", "reply_across_default_dirs", "artifact_yield"],
            "kind": kind,
            "entrypoint": "cli",
            "pidDomain": self._compute_pid_domain(self.registered_pid),
            "messagingSocketPath": self.bridge_socket_path,
            "name": session_name,
            "nameSource": "user",
            "nameSince": now,
            "status": "idle",
            "updatedAt": now,
            "statusUpdatedAt": now
        }

        key_data = {
            "peerToken": self.registered_token,
            "procStart": session_data["procStart"]
        }

        try:
            with open(self.session_json_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
            with open(self.named_descriptor_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)
            with open(self.session_key_path, "w", encoding="utf-8") as f:
                json.dump(key_data, f, indent=2)
            logger.info(f"🚀 Registered bridge session descriptor '{session_name}' (PID {self.registered_pid}) in {self.sessions_dir}")
        except Exception as e:
            logger.error(f"Failed to register session descriptor: {e}")

    def cleanup_session_descriptor(self):
        """
        Cleans up the bridge's registered session descriptor and key files on shutdown.
        """
        for p in [getattr(self, "session_json_path", None), getattr(self, "session_key_path", None), getattr(self, "named_descriptor_path", None)]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                    logger.info(f"Cleaned up session file: {p}")
                except OSError as e:
                    logger.error(f"Failed to remove descriptor file {p}: {e}")

    async def handle_inbound_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """
        Manages inbound response tracking. Listens for external peer confirmations
        and caches payloads for immediate lookup.
        """
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    payload = json.loads(line_str)
                    frame_type = payload.get("type")

                    if frame_type == "auth":
                        logger.info("Inbound peer authenticated successfully.")
                        continue

                    msg_id = payload.get("msg_id", f"inbound_{uuid.uuid4()}")

                    msg_obj = payload.get("message", {})
                    if isinstance(msg_obj, dict):
                        content_val = msg_obj.get("content", payload.get("content", ""))
                    else:
                        content_val = payload.get("content", str(msg_obj))

                    # Real Claude Code senders don't send a bare "sender" name -
                    # they embed identity as from-name inside a
                    # <cross-session-message from=... from-name=... from-mode=...>
                    # wrapper around the actual text, with a top-level "from"
                    # (a "uds:<socket path>" URI) as the only other identity
                    # signal. Only our own older frames (and other bridges
                    # still using the pre-fix format) send a plain "sender"
                    # field, so that's tried first without breaking anything.
                    sender = payload.get("sender")
                    wrapper_match = re.match(
                        r'^<cross-session-message\s+([^>]*)>\n(.*)\n</cross-session-message>$',
                        content_val, re.DOTALL
                    )
                    if wrapper_match:
                        attrs, inner_text = wrapper_match.groups()
                        if sender is None:
                            name_match = re.search(r'from-name="([^"]*)"', attrs)
                            sender = name_match.group(1) if name_match else payload.get("from", "unknown")
                        content_val = inner_text
                    if sender is None:
                        sender = payload.get("from", "unknown")

                    # Cache response
                    self.response_store[msg_id] = {
                        "direction": "inbound",
                        "read": False,
                        "status": payload.get("status", "received"),
                        "sender": sender,
                        "content": content_val,
                        "payload": payload,
                        "timestamp": time.time()
                    }
                    logger.info(f"Buffered inbound response frame: {msg_id}")

                    if self.on_inbound_message:
                        self.on_inbound_message("notifications/message", {
                            "msg_id": msg_id,
                            "sender": sender,
                            "content": content_val,
                            "timestamp": time.time()
                        })

                    ack = json.dumps({"status": "received", "msg_id": msg_id}) + "\n"
                    try:
                        writer.write(ack.encode("utf-8"))
                        await writer.drain()
                    except (ConnectionResetError, BrokenPipeError):
                        pass

                except json.JSONDecodeError:
                    logger.error("Received invalid JSON payload over inbound bridge socket")
                except Exception as ex:
                    logger.error(f"Error processing inbound frame payload: {ex}")
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            logger.info("Inbound peer connection closed.")
        except Exception as e:
            logger.error(f"Error handling inbound traffic: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    def get_unread_messages(self, mark_read: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Returns currently-unread inbound messages (oldest first), the actual
        "check my inbox" affordance for the async-mailbox model: since nothing
        can reliably push/interrupt the connected host, whoever is checking
        needs a way to see only what's new since they last looked, rather than
        re-seeing the same history or having to track msg_ids themselves.
        Marks the returned messages as read unless mark_read is False (to
        peek without consuming).
        """
        unread = {
            msg_id: entry
            for msg_id, entry in self.response_store.items()
            if entry.get("direction") == "inbound" and not entry.get("read", False)
        }
        unread = dict(sorted(unread.items(), key=lambda kv: kv[1].get("timestamp", 0)))
        if mark_read:
            for entry in unread.values():
                entry["read"] = True
        return unread

    async def start_bridge_listener(self):
        """
        Binds an isolated socket interface to aggregate answers and announce
        the server's endpoint to surrounding environments.
        """
        if os.path.exists(self.bridge_socket_path):
            try:
                os.remove(self.bridge_socket_path)
            except OSError as e:
                logger.error(f"Failed to remove stale bridge socket {self.bridge_socket_path}: {e}")

        try:
            server = await asyncio.start_unix_server(
                self.handle_inbound_client, self.bridge_socket_path
            )
            # Apply secure single-user runtime read/write bounds
            os.chmod(self.bridge_socket_path, stat.S_IRUSR | stat.S_IWUSR)
            logger.info(f"🚀 Bridge server operational on: {self.bridge_socket_path}")

            async with server:
                await server.serve_forever()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Bridge listener server error: {e}")

    def cleanup_socket(self):
        if os.path.exists(self.bridge_socket_path):
            try:
                os.remove(self.bridge_socket_path)
                logger.info(f"Cleaned up socket file {self.bridge_socket_path}")
            except OSError as e:
                logger.error(f"Error removing socket file {self.bridge_socket_path}: {e}")
