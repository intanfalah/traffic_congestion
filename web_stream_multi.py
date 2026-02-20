#!/usr/bin/env python3
"""
Web Streaming Server - Multi Feed Support
Supports multiple video sources simultaneously with independent streams.
"""

import os
import cv2
import threading
import time
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime

app = Flask(__name__)

# Dictionary to hold multiple stream handlers
stream_handlers = {}
streams_lock = threading.Lock()


class VideoStream:
    """Individual video stream handler"""
    
    def __init__(self, stream_id, source=0, name="Unnamed"):
        self.stream_id = stream_id
        self.source = source
        self.name = name
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        self.frame_count = 0
        
    def start(self):
        """Start the video stream"""
        if self.running:
            return True
            
        try:
            # Handle different source types
            if isinstance(self.source, str):
                if self.source.isdigit():
                    source = int(self.source)
                else:
                    source = self.source
            else:
                source = self.source
                
            self.capture = cv2.VideoCapture(source)
            
            if not self.capture.isOpened():
                print(f"[Stream {self.stream_id}] Failed to open: {source}")
                return False
                
            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print(f"[Stream {self.stream_id}] Started: {self.name} ({self.source})")
            return True
            
        except Exception as e:
            print(f"[Stream {self.stream_id}] Error starting: {e}")
            return False
    
    def _update(self):
        """Continuously update frames"""
        while self.running:
            if self.capture is not None:
                ret, frame = self.capture.read()
                if ret:
                    current_time = time.time()
                    self.fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time
                    self.frame_count += 1
                    
                    with streams_lock:
                        self.frame = frame
            time.sleep(0.001)
    
    def get_frame(self):
        """Get current frame"""
        with streams_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_jpeg_frame(self):
        """Get current frame as JPEG bytes"""
        frame = self.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                return buffer.tobytes()
        return None
    
    def stop(self):
        """Stop the video stream"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.capture is not None:
            self.capture.release()
        self.capture = None
        print(f"[Stream {self.stream_id}] Stopped")
    
    def is_active(self):
        """Check if stream is active"""
        return self.running and self.capture is not None and self.capture.isOpened()
    
    def get_info(self):
        """Get stream info"""
        return {
            'id': self.stream_id,
            'name': self.name,
            'source': str(self.source),
            'active': self.is_active(),
            'fps': round(self.fps, 1),
            'frames': self.frame_count
        }


def generate_frames(stream_id):
    """Generate MJPEG frames for a specific stream"""
    while True:
        if stream_id in stream_handlers:
            frame_bytes = stream_handlers[stream_id].get_jpeg_frame()
            if frame_bytes is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Main page"""
    return render_template('multi_stream.html')


@app.route('/video_feed/<stream_id>')
def video_feed(stream_id):
    """Video streaming route for specific camera"""
    return Response(generate_frames(stream_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/streams', methods=['GET', 'POST'])
def manage_streams():
    """Manage video streams"""
    global stream_handlers
    
    if request.method == 'GET':
        with streams_lock:
            return jsonify({
                'streams': [s.get_info() for s in stream_handlers.values()]
            })
    
    elif request.method == 'POST':
        data = request.get_json()
        source = data.get('source', 0)
        name = data.get('name', 'Unnamed Stream')
        stream_id = data.get('id', f'stream_{len(stream_handlers) + 1}')
        
        # Stop existing if any
        if stream_id in stream_handlers:
            stream_handlers[stream_id].stop()
            del stream_handlers[stream_id]
        
        # Create new stream
        stream = VideoStream(stream_id, source, name)
        success = stream.start()
        
        if success:
            with streams_lock:
                stream_handlers[stream_id] = stream
            
            port = int(os.environ.get('PORT', 8080))
            return jsonify({
                'success': True,
                'stream': stream.get_info(),
                'url': f'http://127.0.0.1:{port}/video_feed/{stream_id}'
            })
        else:
            return jsonify({
                'success': False,
                'message': f'Failed to start stream from {source}'
            }), 400


@app.route('/api/streams/<stream_id>', methods=['DELETE'])
def delete_stream(stream_id):
    """Delete a stream"""
    global stream_handlers
    
    with streams_lock:
        if stream_id in stream_handlers:
            stream_handlers[stream_id].stop()
            del stream_handlers[stream_id]
            return jsonify({'success': True, 'message': f'Stream {stream_id} deleted'})
        else:
            return jsonify({'success': False, 'message': 'Stream not found'}), 404


@app.route('/api/streams/<stream_id>/stop', methods=['POST'])
def stop_stream(stream_id):
    """Stop a stream"""
    with streams_lock:
        if stream_id in stream_handlers:
            stream_handlers[stream_id].stop()
            return jsonify({'success': True, 'message': f'Stream {stream_id} stopped'})
        else:
            return jsonify({'success': False, 'message': 'Stream not found'}), 404


@app.route('/api/streams/<stream_id>/start', methods=['POST'])
def start_existing_stream(stream_id):
    """Restart a stream"""
    with streams_lock:
        if stream_id in stream_handlers:
            stream = stream_handlers[stream_id]
            success = stream.start()
            return jsonify({
                'success': success,
                'stream': stream.get_info()
            })
        else:
            return jsonify({'success': False, 'message': 'Stream not found'}), 404


@app.route('/api/streams/<stream_id>')
def get_stream_info(stream_id):
    """Get stream info"""
    with streams_lock:
        if stream_id in stream_handlers:
            return jsonify(stream_handlers[stream_id].get_info())
        else:
            return jsonify({'success': False, 'message': 'Stream not found'}), 404


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print("=" * 60)
    print("🎥 Multi-Feed Streaming Server")
    print("=" * 60)
    print(f"\nOpen http://127.0.0.1:{port} in your browser\n")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
