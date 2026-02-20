#!/usr/bin/env python3
"""
Web Streaming Server for Traffic Congestion Analysis
Provides a web interface to view and configure video streams for vehicle detection.
"""

import os
import cv2
import threading
import time
from flask import Flask, render_template, Response, jsonify, request
from datetime import datetime

app = Flask(__name__)

# Global variables for stream management
stream_sources = {
    'camera1': {
        'url': 0,  # Default webcam
        'name': 'Default Camera',
        'active': False,
        'fps': 0
    }
}

current_stream_url = None
stream_capture = None
stream_lock = threading.Lock()


class VideoStream:
    """Thread-safe video stream handler"""
    
    def __init__(self, source=0):
        self.source = source
        self.capture = None
        self.frame = None
        self.running = False
        self.thread = None
        self.fps = 0
        self.last_frame_time = time.time()
        
    def start(self):
        """Start the video stream"""
        if self.running:
            return True
            
        try:
            # Handle different source types
            if isinstance(self.source, str) and self.source.isdigit():
                source = int(self.source)
            else:
                source = self.source
                
            self.capture = cv2.VideoCapture(source)
            
            if not self.capture.isOpened():
                print(f"Failed to open video source: {source}")
                return False
                
            # Set buffer size to reduce latency
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            
            print(f"Video stream started from source: {source}")
            return True
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            return False
    
    def _update(self):
        """Continuously update frames in background thread"""
        while self.running:
            if self.capture is not None:
                ret, frame = self.capture.read()
                if ret:
                    # Calculate FPS
                    current_time = time.time()
                    self.fps = 1.0 / (current_time - self.last_frame_time)
                    self.last_frame_time = current_time
                    
                    with stream_lock:
                        self.frame = frame
            time.sleep(0.001)  # Small delay to prevent CPU overload
    
    def get_frame(self):
        """Get the current frame"""
        with stream_lock:
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
        print("Video stream stopped")
    
    def is_active(self):
        """Check if stream is active"""
        return self.running and self.capture is not None and self.capture.isOpened()


# Initialize global stream instance
current_stream = VideoStream()


def generate_frames():
    """Generate MJPEG frames for streaming"""
    while True:
        frame_bytes = current_stream.get_jpeg_frame()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send placeholder frame if no video
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    """Main page with streaming interface"""
    return render_template('index.html', 
                         sources=stream_sources,
                         current_stream=current_stream_url)


@app.route('/video_feed')
def video_feed():
    """Video streaming route - used by browser and detection system"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stream/start', methods=['POST'])
def start_stream():
    """Start streaming from a specific source"""
    global current_stream_url
    
    data = request.get_json()
    source = data.get('source', 0)
    source_name = data.get('name', 'Unnamed Source')
    
    # Stop current stream if running
    if current_stream.is_active():
        current_stream.stop()
    
    # Update stream source
    current_stream.source = source
    current_stream_url = str(source)
    
    # Start new stream
    success = current_stream.start()
    
    if success:
        # Update sources registry
        stream_id = f"camera{len(stream_sources) + 1}"
        stream_sources[stream_id] = {
            'url': source,
            'name': source_name,
            'active': True,
            'fps': 0
        }
        
        port = int(os.environ.get('PORT', 8080))
        return jsonify({
            'success': True,
            'message': f'Stream started from {source}',
            'stream_id': stream_id,
            'stream_url': f'http://127.0.0.1:{port}/video_feed'
        })
    else:
        return jsonify({
            'success': False,
            'message': f'Failed to start stream from {source}'
        }), 400


@app.route('/api/stream/stop', methods=['POST'])
def stop_stream():
    """Stop the current stream"""
    current_stream.stop()
    return jsonify({'success': True, 'message': 'Stream stopped'})


@app.route('/api/stream/status')
def stream_status():
    """Get current stream status"""
    return jsonify({
        'active': current_stream.is_active(),
        'fps': round(current_stream.fps, 1),
        'source': current_stream_url,
        'sources': stream_sources
    })


@app.route('/api/sources', methods=['GET', 'POST'])
def manage_sources():
    """Manage video sources"""
    global stream_sources
    
    if request.method == 'GET':
        return jsonify(stream_sources)
    
    elif request.method == 'POST':
        data = request.get_json()
        source_id = data.get('id')
        source_url = data.get('url')
        source_name = data.get('name', 'Unnamed Source')
        
        if not source_id or source_url is None:
            return jsonify({'success': False, 'message': 'Missing id or url'}), 400
        
        stream_sources[source_id] = {
            'url': source_url,
            'name': source_name,
            'active': False,
            'fps': 0
        }
        
        return jsonify({
            'success': True,
            'message': f'Source {source_id} added',
            'source': stream_sources[source_id]
        })


@app.route('/api/sources/<source_id>', methods=['DELETE'])
def delete_source(source_id):
    """Delete a video source"""
    global stream_sources
    
    if source_id in stream_sources:
        if source_id != 'camera1':  # Prevent deleting default
            del stream_sources[source_id]
            return jsonify({'success': True, 'message': f'Source {source_id} deleted'})
        else:
            return jsonify({'success': False, 'message': 'Cannot delete default camera'}), 400
    else:
        return jsonify({'success': False, 'message': 'Source not found'}), 404


def get_stream_url():
    """Get the current stream URL for detection system"""
    if current_stream.is_active():
        port = int(os.environ.get('PORT', 8080))
        return f'http://127.0.0.1:{port}/video_feed'
    return None


if __name__ == '__main__':
    # Create templates directory if not exists
    os.makedirs('templates', exist_ok=True)
    
    # Start default stream if available
    print("Starting web streaming server...")
    port = int(os.environ.get('PORT', 8080))
    print(f"Open http://127.0.0.1:{port} in your browser")
    
    # Auto-start with webcam if available
    current_stream.start()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
