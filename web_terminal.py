#!/usr/bin/env python3
"""
Web Terminal Server for AIventure
Runs the terminal app through a web interface using websockets.
"""
import asyncio
import json
import os
import signal
import sys
import threading
import time
import shutil
from pathlib import Path

import pexpect
import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from http.server import HTTPServer, SimpleHTTPRequestHandler
import webbrowser

class WebTerminalServer:
    def __init__(self, command="python launch.py", port=8080, ws_port=8765):
        self.command = command
        self.port = port
        self.ws_port = ws_port
        self.clients = set()
        self.process = None
        self.output_buffer = ""
        self.reader_task = None
        self.lock = asyncio.Lock()

    def start_process(self):
        """Start the AIventure process"""
        try:
            # Set the current working directory to the script's location
            cwd = str(Path(__file__).parent)
            self.process = pexpect.spawn(
                self.command,
                timeout=None,
                encoding='utf-8',
                dimensions=(24, 80),
                cwd=cwd
            )
            self.process.setwinsize(24, 80)
            print(f"Started process: '{self.command}' in '{cwd}'")
            return True
        except Exception as e:
            print(f"Failed to start process: {e}")
            return False

    def stop_process(self):
        """Stop the AIventure process"""
        if self.reader_task:
            self.reader_task.cancel()
            self.reader_task = None
        if self.process and self.process.isalive():
            self.process.terminate()
            self.process = None
    async def broadcast_output(self, output):
        """Append to buffer and send to all connected clients."""
        async with self.lock:
            # Make a copy of clients under the lock to prevent race conditions
            clients_to_send = list(self.clients)
        
        if clients_to_send:
            if output:
                async with self.lock:
                    self.output_buffer += output
                message = json.dumps({'type': 'output', 'data': output})
                await asyncio.gather(*[client.send(message) for client in clients_to_send])

    async def process_reader(self):
        """The single task that reads from the process and broadcasts output."""
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

    async def handle_websocket(self, websocket):
        """Handle a new websocket connection."""
        print(f"New client connected: {websocket.remote_address}")

        async with self.lock:
            if self.process is None:
                if not self.start_process():
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'data': 'Failed to start AIventure process'
                    }))
                    return
                # Start the one and only reader task
                self.reader_task = asyncio.create_task(self.process_reader())
            
            # Add the client to the set and then send the history.
            # This ensures the client is subscribed before getting the history buffer.
            self.clients.add(websocket)
            if self.output_buffer:
                await websocket.send(json.dumps({
                    'type': 'output', 
                    'data': self.output_buffer
                }))

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if not self.process or not self.process.isalive():
                        break

                    if data['type'] == 'input':
                        self.process.send(data['data'])
                    elif data['type'] == 'autocomplete':
                        # Use side-channel socket for autocomplete
                        import socket
                        try:
                            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                                s.settimeout(0.2) # Short timeout for UX
                                s.connect(('127.0.0.1', 9999))
                                s.sendall(data['data'].encode('utf-8'))
                                response = s.recv(4096).decode('utf-8')
                                if response:
                                    res_data = json.loads(response)
                                    if 'suggestions' in res_data:
                                        message = json.dumps({'type': 'autocomplete_results', 'data': res_data['suggestions']})
                                        await websocket.send(message)
                        except (socket.timeout, ConnectionRefusedError):
                            # Silently ignore if autocomplete server is not ready
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
        except (ConnectionClosedError, ConnectionClosedOK):
            pass # Normal disconnection
        finally:
            print(f"Client disconnected: {websocket.remote_address}")
            async with self.lock:
                self.clients.discard(websocket)

    def start_http_server(self):
        """Start HTTP server for serving the web interface"""
        class CustomHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                directory = str(Path(__file__).parent)
                super().__init__(*args, directory=directory, **kwargs)
            
            def log_message(self, format, *args):
                pass
            
        try:
            httpd = HTTPServer(('0.0.0.0', self.port), CustomHandler)
            print(f"HTTP server running on http://0.0.0.0:{self.port}")
            httpd.serve_forever()
        except Exception as e:
            print(f"Failed to start HTTP server: {e}")

    async def start_websocket_server(self):
        """Start WebSocket server"""
        print(f"WebSocket server starting on ws://0.0.0.0:{self.ws_port}")
        try:
            async with websockets.serve(self.handle_websocket, '0.0.0.0', self.ws_port):
                await asyncio.Future()  # Run forever
        except Exception as e:
            print(f"Failed to start WebSocket server: {e}")
            raise

    def run(self):
        """Run the web terminal server"""
        print("Starting AIventure Web Terminal...")
        print(f"Web interface will be available at: http://localhost:{self.port}")
        
        http_thread = threading.Thread(target=self.start_http_server, daemon=True)
        http_thread.start()
        time.sleep(1)
        
        try:
            webbrowser.open(f"http://localhost:{self.port}")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")

        try:
            asyncio.run(self.start_websocket_server())
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.stop_process()

def find_python_executable():
    """Find the correct Python executable"""
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