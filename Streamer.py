import cv2
import socketserver
import http.server
from http import HTTPStatus
from io import BytesIO

class CameraStreamer:
    def __init__(self, camera_index=0, port=5000, resolution=(640, 480)):
        """
        Initialize the camera streamer.
        
        :param camera_index: Index of the camera (e.g., 0 for default camera) or camera URL.
        :param port: Port to run the HTTP server on.
        :param resolution: Resolution of the streamed video (width, height).
        """
        self.camera_index = camera_index
        self.port = port
        self.resolution = resolution
        self.camera = cv2.VideoCapture(self.camera_index)
        self.server = None

    def __del__(self):
        """Release the camera when the object is deleted."""
        if self.camera.isOpened():
            self.camera.release()

    def _generate_frames(self):
        """Generate frames from the camera and encode them as JPEG."""
        while True:
            success, frame = self.camera.read()
            if not success:
                break

            # Resize the frame to the specified resolution
            frame = cv2.resize(frame, self.resolution)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def _handle_request(self, handler):
        """Handle HTTP requests."""
        if handler.path == '/video_feed':
            # Set response headers for streaming
            handler.send_response(HTTPStatus.OK)
            handler.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            handler.end_headers()

            try:
                for frame in self._generate_frames():
                    handler.wfile.write(frame)
            except Exception as e:
                print(f"Streaming error: {e}")
        else:
            # Serve a simple HTML page with the video feed embedded
            handler.send_response(HTTPStatus.OK)
            handler.send_header('Content-Type', 'text/html')
            handler.end_headers()
            handler.wfile.write(f'''
                <html>
                    <head>
                        <title>Camera Stream</title>
                    </head>
                    <body>
                        <h1>Camera Stream</h1>
                        <img src="/video_feed" width="{self.resolution[0]}" height="{self.resolution[1]}">
                    </body>
                </html>
            '''.encode())

    def start(self):
        """Start the HTTP server and stream the camera feed."""
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera.")

        class StreamingHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(handler_self):
                self._handle_request(handler_self)

        class StreamingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self.server = StreamingServer(('0.0.0.0', self.port), StreamingHandler)
        print(f"Starting server on http://0.0.0.0:{self.port}")
        print(f"Access the stream at http://<coprocessor_ip>:{self.port}/video_feed")
        self.server.serve_forever()

    def stop(self):
        """Stop the HTTP server and release the camera."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("Server stopped.")
        if self.camera.isOpened():
            self.camera.release()
            print("Camera released.")

# Example usage
if __name__ == '__main__':
    try:
        # Initialize the streamer with camera index 0, port 5000, and resolution 640x480
        streamer = CameraStreamer(camera_index=0, port=5000, resolution=(640, 480))
        streamer.start()
    except KeyboardInterrupt:
        streamer.stop()