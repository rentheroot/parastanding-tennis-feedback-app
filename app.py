from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
# import mediapipe as mp
import pickle
import os
from datetime import datetime
import json
from config import POSE_LANDMARKS, APP_CONFIG, DEFAULT_PROFILE, ACCESSIBILITY_PRESETS
import main2
app = Flask(__name__)

# Initialize MediaPipe Pose
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
# pose = mp_pose.Pose(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# Global variables
camera = None
current_profile = None
PROFILES_DIR = 'user_profiles'

# Ensure profiles directory exists
if not os.path.exists(PROFILES_DIR):
    os.makedirs(PROFILES_DIR)

# Scenario-specific presets for different tennis conditions
TENNIS_SCENARIOS = {
    'forehand_focus': {
        'name': 'Forehand Training',
        'description': 'Optimized for forehand stroke analysis',
        'enabled_parts': ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW', 
                         'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP']
    },
    'one_arm_right': {
        'name': 'One Arm (Right)',
        'description': 'Right arm disabled - left arm focus',
        'enabled_parts': ['NOSE', 'LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST', 
                         'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
    },
    'one_arm_left': {
        'name': 'One Arm (Left)', 
        'description': 'Left arm disabled - right arm focus',
        'enabled_parts': ['NOSE', 'RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST',
                         'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE']
    },
    'wheelchair': {
        'name': 'Wheelchair Player',
        'description': 'Upper body focus - no leg tracking',
        'enabled_parts': ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 
                         'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST']
    },
    'balance_training': {
        'name': 'Balance Training',
        'description': 'Full body with emphasis on stability',
        'enabled_parts': ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                         'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    },
    'serve_practice': {
        'name': 'Serve Practice',
        'description': 'Optimized for serve motion analysis',
        'enabled_parts': ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                         'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    }
}

def create_scenario_profile(scenario_key):
    """Create profile based on tennis scenario"""
    if scenario_key not in TENNIS_SCENARIOS:
        return None
    
    scenario = TENNIS_SCENARIOS[scenario_key]
    profile = create_default_profile()
    
    # Disable all parts first
    for category in BODY_PARTS.values():
        for part_name in category.keys():
            profile['body_parts'][part_name] = 0
    
    # Enable only scenario-specific parts
    for part_name in scenario['enabled_parts']:
        if any(part_name in category for category in BODY_PARTS.values()):
            profile['body_parts'][part_name] = 1
    
    return profile
BODY_PARTS = {
    'head': {
        'NOSE': 0,
        'LEFT_EYE_INNER': 1,
        'LEFT_EYE': 2,
        'LEFT_EYE_OUTER': 3,
        'RIGHT_EYE_INNER': 4,
        'RIGHT_EYE': 5,
        'RIGHT_EYE_OUTER': 6,
        'LEFT_EAR': 7,
        'RIGHT_EAR': 8,
        'MOUTH_LEFT': 9,
        'MOUTH_RIGHT': 10
    },
    'upper_body': {
        'LEFT_SHOULDER': 11,
        'RIGHT_SHOULDER': 12,
        'LEFT_ELBOW': 13,
        'RIGHT_ELBOW': 14,
        'LEFT_WRIST': 15,
        'RIGHT_WRIST': 16,
        'LEFT_PINKY': 17,
        'RIGHT_PINKY': 18,
        'LEFT_INDEX': 19,
        'RIGHT_INDEX': 20,
        'LEFT_THUMB': 21,
        'RIGHT_THUMB': 22
    },
    'right_hand': {
        'R_WRIST': 0,
        'R_THUMB_CMC': 1,
        'R_THUMB_MCP': 2,
        'R_THUMB_IP': 3,
        'R_THUMB_TIP': 4,
        'R_INDEX_FINGER_MCP': 5,
        'R_INDEX_FINGER_PIP': 6,
        'R_INDEX_FINGER_DIP': 7,
        'R_INDEX_FINGER_TIP': 8,
        'R_MIDDLE_FINGER_MCP': 9,
        'R_MIDDLE_FINGER_PIP': 10,
        'R_MIDDLE_FINGER_DIP': 11,
        'R_MIDDLE_FINGER_TIP': 12,
        'R_RING_FINGER_MCP': 13,
        'R_RING_FINGER_PIP': 14,
        'R_RING_FINGER_DIP': 15,
        'R_RING_FINGER_TIP': 16,
        'R_PINKY_MCP': 17,
        'R_PINKY_PIP': 18,
        'R_PINKY_DIP': 19,
        'R_PINKY_TIP': 20
    },
    'left_hand': {
        'L_WRIST': 0,
        'L_THUMB_CMC': 1,
        'L_THUMB_MCP': 2,
        'L_THUMB_IP': 3,
        'L_THUMB_TIP': 4,
        'L_INDEX_FINGER_MCP': 5,
        'L_INDEX_FINGER_PIP': 6,
        'L_INDEX_FINGER_DIP': 7,
        'L_INDEX_FINGER_TIP': 8,
        'L_MIDDLE_FINGER_MCP': 9,
        'L_MIDDLE_FINGER_PIP': 10,
        'L_MIDDLE_FINGER_DIP': 11,
        'L_MIDDLE_FINGER_TIP': 12,
        'L_RING_FINGER_MCP': 13,
        'L_RING_FINGER_PIP': 14,
        'L_RING_FINGER_DIP': 15,
        'L_RING_FINGER_TIP': 16,
        'L_PINKY_MCP': 17,
        'L_PINKY_PIP': 18,
        'L_PINKY_DIP': 19,
        'L_PINKY_TIP': 20
    },
    'core': {
        'LEFT_HIP': 23,
        'RIGHT_HIP': 24
    },
    'lower_body': {
        'LEFT_KNEE': 25,
        'RIGHT_KNEE': 26,
        'LEFT_ANKLE': 27,
        'RIGHT_ANKLE': 28,
        'LEFT_HEEL': 29,
        'RIGHT_HEEL': 30,
        'LEFT_FOOT_INDEX': 31,
        'RIGHT_FOOT_INDEX': 32
    }
}

def create_default_profile():
    """Create a default profile with all body parts enabled"""
    profile = {
        'username': 'Default User',
        'body_parts': {},
        'created_date': datetime.now().isoformat(),
        'stroke_count': 0,
        'session_history': []
    }
    
    # Set all body parts to enabled (1) by default
    for category in BODY_PARTS.values():
        for part_name, landmark_id in category.items():
            profile['body_parts'][part_name] = 1
    
    return profile

def save_profile(profile):
    """Save user profile to pickle file"""
    filename = f"{profile['username'].replace(' ', '_').lower()}.pkl"
    filepath = os.path.join(PROFILES_DIR, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(profile, f)
    
    return filepath

def load_profile(username):
    """Load user profile from pickle file"""
    filename = f"{username.replace(' ', '_').lower()}.pkl"
    filepath = os.path.join(PROFILES_DIR, filename)
    
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    return None

def get_all_profiles():
    """Get list of all saved profiles"""
    profiles = []
    if os.path.exists(PROFILES_DIR):
        for filename in os.listdir(PROFILES_DIR):
            if filename.endswith('.pkl'):
                filepath = os.path.join(PROFILES_DIR, filename)
                try:
                    with open(filepath, 'rb') as f:
                        profile = pickle.load(f)
                        profiles.append({
                            'username': profile['username'],
                            'created_date': profile.get('created_date', 'Unknown'),
                            'stroke_count': profile.get('stroke_count', 0)
                        })
                except:
                    continue
    return profiles

def generate_frames():
    """Generate frames for video streaming with simple camera feed"""
    global camera, current_profile
    # Start detector feed in background and stream annotated frames
    vf = main2.VideoFeed()
    vf.start_feed(cam_index=0)

    # run the detection loop in a daemon thread (headless)
    t = threading.Thread(target=vf.trace_body_pos, args=(True,), daemon=True)
    t.start()

    # stream latest annotated frame from the detector's buffer
    while True:
        try:
            if vf.frame_buffer_annot:
                ts, frame = vf.frame_buffer_annot[-1]
                # overlay current profile info if available
                if current_profile:
                    cv2.putText(frame, f"User: {current_profile['username']}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"Strokes: {current_profile['stroke_count']}",
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # no frame yet, wait briefly
                time.sleep(0.01)
                continue
        except GeneratorExit:
            break
        except Exception:
            time.sleep(0.01)
            continue

@app.route('/')
def index():
    """Landing page - Profile management"""
    profiles = get_all_profiles()
    return render_template('landing.html', 
                         body_parts=BODY_PARTS,
                         profiles=profiles)

@app.route('/monitor')
def monitor():
    """Monitoring page - Real-time video tracking"""
    global current_profile
    if current_profile is None:
        return "No profile selected. Please create or select a profile first.", 400
    return render_template('monitor.html', profile=current_profile)

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/profile/create', methods=['POST'])
def create_profile():
    """API endpoint to create new profile"""
    data = request.json
    profile = create_default_profile()
    profile['username'] = data['username']
    profile['body_parts'] = data['body_parts']
    
    save_profile(profile)
    return jsonify({'status': 'success', 'message': 'Profile created successfully'})

@app.route('/api/scenario/create', methods=['POST'])
def create_scenario_profile_api():
    """Create profile based on tennis scenario"""
    data = request.json
    username = data['username']
    scenario_key = data['scenario']
    
    profile = create_scenario_profile(scenario_key)
    if profile:
        profile['username'] = username
        profile['scenario'] = scenario_key
        save_profile(profile)
        return jsonify({'status': 'success', 'profile': profile})
    return jsonify({'status': 'error', 'message': 'Invalid scenario'}), 400

@app.route('/api/scenarios', methods=['GET'])
def get_scenarios():
    """Get available tennis scenarios"""
    return jsonify({'scenarios': TENNIS_SCENARIOS})

@app.route('/api/profile/load', methods=['POST'])
def load_profile_api():
    """API endpoint to load existing profile"""
    global current_profile
    data = request.json
    username = data['username']
    
    profile = load_profile(username)
    if profile:
        current_profile = profile
        return jsonify({'status': 'success', 'profile': profile})
    return jsonify({'status': 'error', 'message': 'Profile not found'}), 404

@app.route('/api/profile/delete', methods=['POST'])
def delete_profile():
    """API endpoint to delete profile"""
    data = request.json
    username = data['username']
    
    filename = f"{username.replace(' ', '_').lower()}.pkl"
    filepath = os.path.join(PROFILES_DIR, filename)
    
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'status': 'success', 'message': 'Profile deleted successfully'})
    return jsonify({'status': 'error', 'message': 'Profile not found'}), 404

@app.route('/api/stroke/increment', methods=['POST'])
def increment_stroke():
    """API endpoint to increment stroke count (called by Arduino button)"""
    global current_profile
    
    if current_profile:
        current_profile['stroke_count'] += 1
        save_profile(current_profile)
        return jsonify({'status': 'success', 'count': current_profile['stroke_count']})
    return jsonify({'status': 'error'}), 400

@app.route('/api/config/export', methods=['GET'])
def export_config():
    """Export current profile config for ML teammate"""
    global current_profile
    
    if current_profile:
        # Create config in format teammate expects
        config = {
            'username': current_profile['username'],
            'enabled_pose_landmarks': [],
            'body_parts_config': current_profile['body_parts']
        }
        
        # Add enabled landmark names to list
        for part_name, enabled in current_profile['body_parts'].items():
            if enabled == 1:
                config['enabled_pose_landmarks'].append(part_name)
        
        return jsonify(config)
    return jsonify({'status': 'error', 'message': 'No active profile'}), 400

@app.route('/api/presets', methods=['GET'])
def get_presets():
    """Get accessibility presets from config"""
    return jsonify({'presets': ACCESSIBILITY_PRESETS})

@app.route('/stop')
def stop():
    """Stop camera and return to landing page"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return render_template('landing.html', body_parts=BODY_PARTS)

if __name__ == '__main__':
    app.run(debug=APP_CONFIG['debug'], host=APP_CONFIG['host'], port=APP_CONFIG['port'])
