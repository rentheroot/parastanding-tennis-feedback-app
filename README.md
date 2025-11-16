# 🎾 Parastanding Tennis Coach

An inclusive tennis training application that uses AI-powered pose analysis to help players with disabilities improve their forehand technique.

## 🚀 Quick Start
### 1. Create & activate a virtual environment (recommended)
If you already have a virtualenv you can reuse it, otherwise create one (example using Python 3.11 which is known to be well supported for MediaPipe):

```bash
# create (or replace) the venv at tennis_env using a compatible Python
python3.11 -m venv tennis_env
source tennis_env/bin/activate

# upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel
```

> Note: MediaPipe ships platform-specific binary wheels. If pip reports "No matching distribution found for mediapipe" on your Python version, recreate the venv with Python 3.11 (recommended) or try 3.12.12. See the Troubleshooting section below.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser
- Landing page: `http://localhost:5001/`
- Monitoring page (after selecting a profile): `http://localhost:5001/monitor`

Note: the Flask server binds to the address and port defined in `config.py` (defaults: host `0.0.0.0`, port `5001`). When running in debug mode Flask prints the development server warning and accessible URLs, for example:

```
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5001
 * Running on http://10.110.201.39:5001
```

If you want to expose the app only locally, change `APP_CONFIG['host']` to `127.0.0.1` in `config.py` or run via a production WSGI server (Gunicorn / uWSGI) for deployment.

## 📱 Features

### Landing Page
- **User Profile Management**: Create and save player profiles
- **Accessibility Presets**: 
  - One Arm (Right/Left)
  - Wheelchair Player
  - Full Body Tracking
- **Custom Body Part Selection**: Choose which body parts to track
- **Profile Persistence**: Saves configurations using pickle files
 - **Profile Delete (UI)**: Each saved profile has a Delete control in the UI; this calls `/api/profile/delete` (frontend only). Implement the small backend route to make deletion persistent (instructions below).
 - **Improved Start Training UX**: Start Training will auto-select the first saved profile, auto-select a matching profile when you type a username, enable the Start button when a username/profile is present, and will automatically create a missing profile when you click Start.

### Monitoring Page
- **Real-time Pose Tracking**: Live camera feed with MediaPipe overlay
- **Session Statistics**: Stroke count, session time, performance metrics
- **Arduino Integration**: Ready for hardware button input
- **AI Recommendations**: Placeholder for ML model suggestions
- **Keyboard Shortcuts**: Spacebar (stroke), P (pause), R (reset), Escape (stop)
Note: MediaPipe is optional at runtime — the app now guards the import and will run without MediaPipe installed (useful during development). To enable full pose tracking install MediaPipe in a venv created with a compatible Python as described above.

## 🔧 Integration Points

### For ML Teammate
The pose tracking provides 33 MediaPipe landmarks perfect for tennis analysis:

```python
# In generate_frames() function - line 89
if results.pose_landmarks:
    # Access pose landmarks here
    landmarks = results.pose_landmarks.landmark
    # Feed into your stroke detection model
```

**Key landmarks for tennis:**
- Shoulders (11, 12)
- Elbows (13, 14) 
- Wrists (15, 16)
- Hips (23, 24)

### For Arduino Teammate
Ready-to-use API endpoint for button integration:

```bash
# Test stroke increment
curl -X POST http://localhost:5001/api/stroke/increment
```

**Arduino Integration Steps:**
1. Connect Arduino to WiFi
2. Send POST request to `/api/stroke/increment` when button pressed
3. Accelerometer data can be sent to `/api/sensor/data` (implement as needed)

## 🌐 Cloudflare Deployment

### Using Cloudflare Tunnel
```bash
# Install cloudflared
brew install cloudflare/cloudflare/cloudflared

# Create tunnel
cloudflared tunnel create tennis-coach

# Run tunnel
cloudflared tunnel --url http://localhost:5000
```

### Cloudflare Workers Integration
- Video streaming can use Cloudflare Stream
- Static assets deployable to R2
- ML inference via Workers AI

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Landing page |
| `/monitor` | GET | Training monitor |
| `/video_feed` | GET | Video stream |
| `/api/profile/create` | POST | Create new profile |
| `/api/profile/load` | POST | Load existing profile |
| `/api/profile/delete` | POST | (UI calls this) Delete profile (backend route not implemented by default) |
| `/api/stroke/increment` | POST | Increment stroke count |
| `/stop` | GET | Stop camera and return home |

## 🎯 Accessibility Features

### For Players with Disabilities
- **Customizable Tracking**: Enable/disable specific body parts
- **One-Arm Presets**: Optimized for single-arm players
- **Wheelchair Support**: Upper-body focused tracking
- **High Contrast UI**: Colorblind-friendly design
- **Large Touch Targets**: Easy interaction on all devices

### Technical Accessibility
- **Keyboard Navigation**: Full app control via keyboard
- **Screen Reader Support**: ARIA labels throughout
- **Responsive Design**: Works on tablets and phones
- **Clear Visual Feedback**: Status indicators and animations

### Medical Terminology
The following terms are the preferred clinical/medical names for common amputation levels referenced in the app and documentation. Use these terms when communicating with healthcare or rehabilitation professionals.

- Transradial (below elbow) — forearm amputation (amputation distal to the elbow joint; preserves elbow function)
- Transhumeral (above elbow) — upper arm amputation (amputation proximal to the elbow; elbow joint is absent)
- Transtibial (below knee) — lower leg amputation (amputation distal to the knee joint; preserves knee function)
- Transfemoral (above knee) — upper leg amputation (amputation proximal to the knee; knee joint is absent)

These terms describe anatomical level and are widely used in prosthetics, rehabilitation, and clinical documentation. When appropriate, pair the term with side (e.g., "left transradial amputation") and functional notes (e.g., preserved elbow ROM).

### Professional description guidance
- Use clear anatomical references: specify level (e.g., transradial vs transtibial), side (left/right) and whether it is partial or complete.
- Proper medical classification: prefer the standard surgical/rehabilitation terminology above rather than colloquial phrases (for example, "below-elbow amputation" → "transradial").
- Suitable for healthcare/rehabilitation settings: include preserved joints and functional implications (e.g., "transradial — preserves elbow flexion/extension; may use terminal device or myoelectric prosthesis").


## 🔬 Technical Architecture

```
parastanding-tennis-feedback-app/
├── app.py                 # Flask application
├── templates/
│   ├── landing.html       # Profile management
│   └── monitor.html       # Training interface
├── user_profiles/         # Saved profiles (auto-created)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎮 Demo Script for Judges

1. **Show Accessibility**:
   - Create profile with "One Arm (Right)" preset
   - Demonstrate how it only tracks left side
   - Explain benefit for amputee athletes

2. **Show Real-time Tracking**:
   - Perform tennis swing motions
   - Point out pose landmarks overlay
   - Use spacebar to increment stroke counter

3. **Show Profile System**:
   - Create multiple profiles (wheelchair, full body)
   - Switch between them instantly
   - Show data persistence

4. **Mention Integration**:
   - "ML model will analyze form here"
   - "Arduino will detect swing speed"
   - "Cloudflare Workers for edge AI"

## 🛠 Development Notes

### Current Implementation
- ✅ MediaPipe pose detection
- ✅ Flask web interface
- ✅ Profile management with pickle
- ✅ Real-time video streaming
- ✅ Accessibility presets
- ✅ Arduino API endpoints

### Next Steps for Team
1. **ML Integration**: Connect stroke detection model to `generate_frames()`
2. **Arduino Hardware**: Implement accelerometer + button
3. **Cloudflare Deploy**: Set up tunnel and Workers
4. **Enhanced UI**: Add more visual feedback and animations

## 🏥 Health & Safety
- Camera permission required
- No personal data stored beyond local profiles
- All processing happens locally for privacy
- Suitable for rehabilitation and training environments

## 📞 Support
For hackathon questions or technical issues, check the console logs or create an issue in the repository.

---
**Built for Technica 2025 Hackathon** 🏆
