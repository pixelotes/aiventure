#!/usr/bin/env python3
"""
Web Terminal Server for AIventure.
Serves the static UI and the WebSocket on the same port (default 8080).
The WebSocket endpoint lives at /ws — the browser uses location.host so it
works locally, in Docker, behind an Ingress, with or without TLS.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
import asyncio
import json
import shutil
import socket
from pathlib import Path

import pexpect
from aiohttp import web, WSMsgType


class WebTerminalServer:
    def __init__(self, command="python cli.py", port=8080):
        self.command = command
        self.port = port
        self.clients = set()
        self.process = None
        self.output_buffer = ""
        self.reader_task = None
        self.lock = asyncio.Lock()

    def start_process(self):
        try:
            cwd = str(Path(__file__).parent)
            self.process = pexpect.spawn(
                self.command,
                timeout=None,
                encoding='utf-8',
                dimensions=(24, 80),
                cwd=cwd,
            )
            self.process.setwinsize(24, 80)
            print(f"Started process: '{self.command}' in '{cwd}'")
            return True
        except Exception as e:
            print(f"Failed to start process: {e}")
            return False

    def stop_process(self):
        if self.reader_task:
            self.reader_task.cancel()
            self.reader_task = None
        if self.process and self.process.isalive():
            self.process.terminate()
            self.process = None

    async def broadcast_output(self, output):
        async with self.lock:
            clients_to_send = list(self.clients)
        if clients_to_send and output:
            async with self.lock:
                self.output_buffer += output
            message = json.dumps({'type': 'output', 'data': output})
            await asyncio.gather(*[c.send_str(message) for c in clients_to_send])

    async def process_reader(self):
        print("Process reader started.")
        while True:
            await self.broadcast_output("Starting AIventure... please wait.\r\n")
            if not self.process or not self.process.isalive():
                if not self.start_process():
                    await self.broadcast_output("\r\n[Error: Failed to start process]\r\n")
                    break

            while self.process and self.process.isalive():
                try:
                    output = self.process.read_nonblocking(size=1024, timeout=0.1)
                    if output:
                        await self.broadcast_output(output)
                except pexpect.TIMEOUT:
                    await asyncio.sleep(0.01)
                except pexpect.EOF:
                    break

            await self.broadcast_output('\r\n[Process ended - Restarting in 3s...]\r\n')
            print("Process ended. Waiting to restart...")
            await asyncio.sleep(3)

    async def handle_websocket(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        peer = request.remote
        print(f"New client connected: {peer}")

        async with self.lock:
            if self.process is None:
                if not self.start_process():
                    await ws.send_str(json.dumps({
                        'type': 'error',
                        'data': 'Failed to start AIventure process',
                    }))
                    return ws
                self.reader_task = asyncio.create_task(self.process_reader())
            self.clients.add(ws)
            if self.output_buffer:
                await ws.send_str(json.dumps({
                    'type': 'output',
                    'data': self.output_buffer,
                }))

        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                    if not self.process or not self.process.isalive():
                        break
                    if data['type'] == 'input':
                        self.process.send(data['data'])
                    elif data['type'] == 'autocomplete':
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.settimeout(0.2)
                                s.connect(('127.0.0.1', 9999))
                                s.sendall(data['data'].encode('utf-8'))
                                response = s.recv(4096).decode('utf-8')
                                if response:
                                    res_data = json.loads(response)
                                    if 'suggestions' in res_data:
                                        await ws.send_str(json.dumps({
                                            'type': 'autocomplete_results',
                                            'data': res_data['suggestions'],
                                        }))
                        except (socket.timeout, ConnectionRefusedError):
                            pass
                        except Exception as e:
                            print(f"Autocomplete side-channel error: {e}")
                    elif data['type'] == 'resize':
                        rows = data.get('rows', 24)
                        cols = data.get('cols', 80)
                        self.process.setwinsize(rows, cols)
                except json.JSONDecodeError:
                    print("Invalid JSON received")
                except Exception as e:
                    print(f"Error handling client message: {e}")
        finally:
            print(f"Client disconnected: {peer}")
            async with self.lock:
                self.clients.discard(ws)
        return ws

    async def index(self, request):
        return web.FileResponse(Path(__file__).parent / "index.html")

    def build_app(self):
        app = web.Application()
        app.router.add_get('/', self.index)
        app.router.add_get('/ws', self.handle_websocket)
        # Static files (CSS/JS/images) served from the script dir.
        app.router.add_static('/', path=str(Path(__file__).parent), show_index=False)
        return app

    def run(self):
        print("Starting AIventure Web Terminal...")
        print(f"HTTP + WS available at http://0.0.0.0:{self.port}  (WS path: /ws)")
        try:
            web.run_app(self.build_app(), host='0.0.0.0', port=self.port, access_log=None)
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_process()


def find_python_executable():
    python_names = ['python3', 'python']
    for name in python_names:
        if shutil.which(name):
            return name
    return sys.executable


if __name__ == "__main__":
    python_exec = find_python_executable()
    command = f"{python_exec} ./cli.py"
    if len(sys.argv) > 1:
        command = " ".join(sys.argv[1:])
    print(f"Using command: {command}")
    server = WebTerminalServer(command=command)
    server.run()
