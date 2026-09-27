#!/usr/bin/env python3
"""
Comprehensive Test Suite for Claude Message Bridge MCP Server (claude-mcp2)
"""
import os
import sys
import json
import time
import glob
import uuid
import asyncio
import tempfile
import unittest
import subprocess

# Ensure repo directory is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from claude_chatter import ClaudeMessagingProtocol


class TestBridgeSessionManagement(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.tmp_dir.name, "test_session_mgmt.sock")
        self.bridge = ClaudeMessagingProtocol(bridge_socket_path=self.socket_path)

    async def asyncTearDown(self):
        self.bridge.cleanup_session_descriptor()
        self.bridge.cleanup_socket()
        self.tmp_dir.cleanup()

    async def test_session_descriptor_registration(self):
        """Verify session descriptor and key file creation & cleanup."""
        self.bridge.register_session_descriptor(session_name="test-bridge-suite")

        # Check files exist
        self.assertTrue(os.path.exists(self.bridge.session_json_path))
        self.assertTrue(os.path.exists(self.bridge.session_key_path))

        # Check json content
        with open(self.bridge.session_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["name"], "test-bridge-suite")
            self.assertEqual(data["messagingSocketPath"], self.socket_path)

        # Check key content
        with open(self.bridge.session_key_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)
            self.assertEqual(key_data["peerToken"], self.bridge.registered_token)

        # Cleanup
        self.bridge.cleanup_session_descriptor()
        self.assertFalse(os.path.exists(self.bridge.session_json_path))
        self.assertFalse(os.path.exists(self.bridge.session_key_path))

    async def test_session_descriptor_pid_fields_match_real_kernel_values(self):
        """Verify pidDomain/procStart are the real kernel-reported values a
        native Claude Code session writes (machine-id + PID-namespace inode;
        /proc/<pid>/stat's starttime field), not placeholders derived from
        the literal pid - a real session's descriptor was observed to reject
        cross-session delivery to a bridge using the old placeholder shape."""
        self.bridge.register_session_descriptor(session_name="test-pid-fields")
        try:
            with open(self.bridge.session_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            try:
                with open("/etc/machine-id", "r", encoding="utf-8") as f:
                    machine_id = f.read().strip()
            except OSError:
                machine_id = "local"  # e.g. inside this project's own minimal test container
            ns_part = os.readlink(f"/proc/{os.getpid()}/ns/pid").split(":", 1)[1]
            self.assertEqual(data["pidDomain"], f"linux:{machine_id}:pid:{ns_part}")

            with open(f"/proc/{os.getpid()}/stat", "r", encoding="utf-8") as f:
                expected_proc_start = f.read().rsplit(")", 1)[1].split()[19]
            self.assertEqual(data["procStart"], expected_proc_start)
            self.assertNotEqual(data["procStart"], str(os.getpid()))

            with open(self.bridge.session_key_path, "r", encoding="utf-8") as f:
                key_data = json.load(f)
            self.assertEqual(key_data["procStart"], data["procStart"])
        finally:
            self.bridge.cleanup_session_descriptor()

    async def test_dead_session_purging(self):
        """Verify purging of dead session descriptors and key files."""
        sessions_dir = self.bridge.sessions_dir
        os.makedirs(sessions_dir, exist_ok=True)

        fake_pid = 9999999
        fake_json = os.path.join(sessions_dir, f"{fake_pid}.json")
        fake_key = os.path.join(sessions_dir, f"{fake_pid}.abcd1234key.key")

        with open(fake_json, "w", encoding="utf-8") as f:
            json.dump({"pid": fake_pid, "messagingSocketPath": "/tmp/fake.sock", "name": "dead-session"}, f)

        with open(fake_key, "w", encoding="utf-8") as f:
            json.dump({"peerToken": "fake_token", "procStart": str(fake_pid)}, f)

        self.assertTrue(os.path.exists(fake_json))
        self.assertTrue(os.path.exists(fake_key))

        # Run verification and purging
        await self.bridge.verify_and_purge_sessions()

        self.assertFalse(os.path.exists(fake_json))
        self.assertFalse(os.path.exists(fake_key))


class TestUnreadMessages(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.bridge = ClaudeMessagingProtocol(bridge_socket_path=os.path.join(self.tmp_dir.name, "test.sock"))

    async def asyncTearDown(self):
        self.tmp_dir.cleanup()

    async def test_get_unread_messages_marks_read_and_excludes_outbound(self):
        """Verify get_unread_messages returns only unread inbound entries, marks
        them read by default, and never surfaces our own outbound dispatch
        confirmations (which share the same response_store but aren't inbox
        items)."""
        self.bridge.response_store["out_1"] = {
            "direction": "outbound", "status": "dispatched", "content": "hi"
        }
        self.bridge.response_store["in_1"] = {
            "direction": "inbound", "read": False, "sender": "peer-a",
            "content": "first", "timestamp": 1.0
        }
        self.bridge.response_store["in_2"] = {
            "direction": "inbound", "read": False, "sender": "peer-b",
            "content": "second", "timestamp": 2.0
        }

        unread = self.bridge.get_unread_messages()
        self.assertEqual(list(unread.keys()), ["in_1", "in_2"])
        self.assertTrue(self.bridge.response_store["in_1"]["read"])
        self.assertTrue(self.bridge.response_store["in_2"]["read"])

        # A second call should now return nothing - already marked read.
        self.assertEqual(self.bridge.get_unread_messages(), {})

    async def test_get_unread_messages_peek_without_consuming(self):
        """Verify mark_read=False lets a caller peek without consuming."""
        self.bridge.response_store["in_1"] = {
            "direction": "inbound", "read": False, "sender": "peer-a",
            "content": "first", "timestamp": 1.0
        }
        unread = self.bridge.get_unread_messages(mark_read=False)
        self.assertEqual(list(unread.keys()), ["in_1"])
        self.assertFalse(self.bridge.response_store["in_1"]["read"])
        # still there on the next real read
        self.assertEqual(list(self.bridge.get_unread_messages().keys()), ["in_1"])


class TestBridgeIPCCommunication(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.tmp_dir.name, "test_ipc.sock")
        self.bridge = ClaudeMessagingProtocol(bridge_socket_path=self.socket_path)

    async def asyncTearDown(self):
        self.bridge.cleanup_session_descriptor()
        self.bridge.cleanup_socket()
        self.tmp_dir.cleanup()

    async def test_inbound_listener_and_buffering(self):
        """Verify inbound Unix socket listener receives and buffers peer frames."""
        server_task = asyncio.create_task(self.bridge.start_bridge_listener())
        await asyncio.sleep(0.1)

        self.assertTrue(os.path.exists(self.socket_path))

        reader, writer = await asyncio.open_unix_connection(self.socket_path)

        # Send auth frame
        auth_frame = {"type": "auth", "peerToken": "test_token_123"}
        writer.write((json.dumps(auth_frame) + "\n").encode("utf-8"))

        # Send message frame
        msg_id = "test_msg_suite_99"
        msg_frame = {
            "type": "user",
            "message": {"role": "user", "content": "Suite Test Payload"},
            "msg_id": msg_id,
            "sender": "suite_agent"
        }
        writer.write((json.dumps(msg_frame) + "\n").encode("utf-8"))
        await writer.drain()

        # Read ACK
        ack_line = await reader.readline()
        ack_data = json.loads(ack_line.decode("utf-8"))
        self.assertEqual(ack_data.get("status"), "received")
        self.assertEqual(ack_data.get("msg_id"), msg_id)

        writer.close()
        await writer.wait_closed()

        # Verify buffered payload in response_store
        self.assertIn(msg_id, self.bridge.response_store)
        cached = self.bridge.response_store[msg_id]
        self.assertEqual(cached["sender"], "suite_agent")
        self.assertEqual(cached["content"], "Suite Test Payload")

        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

    async def test_outbound_send_to_mock_peer(self):
        """Verify outbound frame formatting and delivery over target Unix domain socket."""
        mock_sock_path = os.path.join(self.tmp_dir.name, "mock_target.sock")
        received_frames = []

        async def mock_handler(reader, writer):
            while True:
                line = await reader.readline()
                if not line:
                    break
                received_frames.append(json.loads(line.decode("utf-8").strip()))
            writer.close()
            await writer.wait_closed()

        mock_server = await asyncio.start_unix_server(mock_handler, mock_sock_path)

        self.bridge.active_peers = {
            "mock-peer": {
                "name": "mock-peer",
                "pid": 88888,
                "sessionId": "mock-session-xyz",
                "socket": mock_sock_path,
                "token": "secret_peer_token",
                "cwd": "/tmp"
            }
        }

        async def bypass_verify():
            return self.bridge.active_peers
        self.bridge.verify_and_purge_sessions = bypass_verify

        res = await self.bridge.send_to_claude("mock-peer", "Hello Mock Peer!")
        await asyncio.sleep(0.05)

        self.assertTrue(res["success"])
        self.assertEqual(len(received_frames), 2)
        self.assertEqual(received_frames[0]["type"], "auth")
        self.assertEqual(received_frames[0]["peerToken"], "secret_peer_token")
        self.assertEqual(received_frames[1]["type"], "user")
        self.assertEqual(received_frames[1]["message"]["content"], "Hello Mock Peer!")

        mock_server.close()
        await mock_server.wait_closed()


class TestMCPStdioProtocol(unittest.TestCase):
    def test_stdio_rpc(self):
        """Verify standard JSON-RPC 2.0 stdio MCP methods."""
        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bridge_mcp.py"))
        socket_path = f"/tmp/test_rpc_{uuid.uuid4().hex[:8]}.sock"

        # The stdio process's default session name is derived from cwd - run it from
        # an isolated temp directory, not this repo's own directory, so it can never
        # collide with (or overwrite the identity of) a real developer session for
        # this same repo.
        test_cwd = tempfile.mkdtemp(prefix="bridge_mcp_test_")

        proc = subprocess.Popen(
            [sys.executable, script_path, "--socket", socket_path],
            cwd=test_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            # 1. initialize - FastMCP validates against the real MCP spec, which
            # requires protocolVersion/capabilities/clientInfo (unlike the old
            # hand-rolled server, which accepted anything here).
            init_req = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-suite", "version": "1.0"}
                }
            }
            proc.stdin.write(json.dumps(init_req) + "\n")
            proc.stdin.flush()
            init_resp = json.loads(proc.stdout.readline())
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "claudemessaging")

            # 2. ping
            ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}}
            proc.stdin.write(json.dumps(ping_req) + "\n")
            proc.stdin.flush()
            ping_resp = json.loads(proc.stdout.readline())
            self.assertEqual(ping_resp["result"], {})

            # 3. tools/list
            list_req = {"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}}
            proc.stdin.write(json.dumps(list_req) + "\n")
            proc.stdin.flush()
            list_resp = json.loads(proc.stdout.readline())
            tool_names = [t["name"] for t in list_resp["result"]["tools"]]
            # 4. tools/call list_sessions
            call_req = {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": {}}
            }
            proc.stdin.write(json.dumps(call_req) + "\n")
            proc.stdin.flush()
            call_resp = json.loads(proc.stdout.readline())
            self.assertFalse(call_resp["result"]["isError"])
            self.assertIn("content", call_resp["result"])

            # 5. tools/call rename_session
            rename_req = {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "rename_session", "arguments": {"new_name": "renamed-suite-bridge"}}
            }
            proc.stdin.write(json.dumps(rename_req) + "\n")
            proc.stdin.flush()
            rename_resp = json.loads(proc.stdout.readline())
            self.assertFalse(rename_resp["result"]["isError"])
            rename_data = json.loads(rename_resp["result"]["content"][0]["text"])
            self.assertEqual(rename_data["new_name"], "renamed-suite-bridge")

        finally:
            proc.terminate()
            proc.wait(timeout=5)
            if os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                except OSError:
                    pass
            try:
                os.rmdir(test_cwd)
            except OSError:
                pass


class TestBridgeStandaloneCLI(unittest.TestCase):
    def setUp(self):
        self.script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bridge_mcp.py"))

    def test_cli_list(self):
        """Verify python3 bridge_mcp.py --list executes and returns valid JSON."""
        res = subprocess.run(
            [sys.executable, self.script_path, "--list"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        data = json.loads(res.stdout)
        self.assertIsInstance(data, dict)

    def test_cli_purge(self):
        """Verify python3 bridge_mcp.py --purge executes successfully."""
        res = subprocess.run(
            [sys.executable, self.script_path, "--purge"],
            capture_output=True,
            text=True
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Purge scan completed", res.stdout)

    def test_cli_standalone_daemon(self):
        """Verify python3 bridge_mcp.py --standalone starts and stops gracefully on SIGTERM."""
        socket_path = f"/tmp/test_standalone_{uuid.uuid4().hex[:8]}.sock"
        proc = subprocess.Popen(
            [sys.executable, self.script_path, "--standalone", "--socket", socket_path, "--name", "test-standalone-daemon"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            time.sleep(1.0)
            self.assertTrue(os.path.exists(socket_path))
            proc.terminate()
            stdout, stderr = proc.communicate(timeout=3)
            self.assertEqual(proc.returncode, 0)
            self.assertIn("Bridge standalone daemon shut down gracefully", stderr)
            self.assertFalse(os.path.exists(socket_path))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                except OSError:
                    pass

    def test_cli_standalone_send(self):
        """Verify python3 bridge_mcp.py --send dispatches to a standalone daemon.
        send_to_claude is fire-and-forget (no auto-reply scaffold, no transcript
        polling) - a real reply, if any, arrives later via a genuine send_message
        call from the target instead."""
        socket_path = f"/tmp/test_pingpong_{uuid.uuid4().hex[:8]}.sock"
        daemon_name = f"suite-daemon-{uuid.uuid4().hex[:6]}"

        proc = subprocess.Popen(
            [sys.executable, self.script_path, "--standalone", "--socket", socket_path, "--name", daemon_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            time.sleep(1.0)
            self.assertTrue(os.path.exists(socket_path))

            # Send message to the daemon using --send; dispatch should succeed
            # immediately since send_to_claude is fire-and-forget.
            res = subprocess.run(
                [sys.executable, self.script_path, "--send", daemon_name, "hello standalone"],
                capture_output=True,
                text=True,
                timeout=10
            )
            self.assertEqual(res.returncode, 0)
            data = json.loads(res.stdout)
            self.assertTrue(data.get("success"))
            self.assertEqual(data.get("status"), "dispatched")

        finally:
            proc.terminate()
            proc.wait(timeout=3)
            if os.path.exists(socket_path):
                try:
                    os.remove(socket_path)
                except OSError:
                    pass


if __name__ == "__main__":
    unittest.main()
