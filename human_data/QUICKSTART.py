"""
Quick start guide for the LBF Human Interaction Web App
"""

print("""
╔══════════════════════════════════════════════════════════════╗
║     LBF Human Interaction Web Application - Quick Start      ║
╚══════════════════════════════════════════════════════════════╝

📁 Files Created:
   ├── app.py                  - Flask backend server
   ├── templates/index.html    - Web interface
   ├── README.md              - Full documentation
   ├── requirements.txt       - Dependencies
   ├── start_server.sh        - Easy startup script
   ├── test_app.py            - Test components
   └── SUMMARY.md             - This summary

🎮 Controls:
   W/↑ : Move Up        A/← : Move Left
   S/↓ : Move Down      D/→ : Move Right
   SPACE : Collect Food      Q : Wait

🚀 Quick Start:

   1. Install dependencies:
      $ pip install flask flask-cors

   2. Start the server:
      $ cd /scratch/cluster/jyliu/Documents/jax-aht/human_data
      $ ./start_server.sh
      
      (or simply: python app.py)

   3. Open your browser:
      http://localhost:5000

   4. Play the game - episodes auto-save when complete!

📊 Data Collection:
   - Episodes AUTOMATICALLY saved to: collected_data/
   - Saved when: Episode completes (max steps or game ends)
   - Format: JSON with full trajectory + timestamp
   - Includes: actions, rewards, states, session info

🎯 Game Objective:
   You (👤 red) and AI (🤖 teal) work together to collect
   food (🍎). You can only collect food if your level is
   high enough. Maximize team score!

🔧 Architecture:
   
   ┌─────────────────┐
   │   Web Browser   │  ← You play here
   │  (JavaScript)   │
   └────────┬────────┘
            │ HTTP/JSON
   ┌────────▼────────┐
   │  Flask Server   │  ← Backend (app.py)
   │   (Python)      │
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │  JAX-AHT Env    │  ← LBF Environment
   │  + Heuristic    │     + AI Agent
   │     Agent       │
   └─────────────────┘

📚 For More Details:
   - Read README.md for full documentation
   - Read SUMMARY.md for architecture overview
   - Run test_app.py to verify setup

🤝 Need Help?
   - Check README.md troubleshooting section
   - Make sure conda environment is activated
   - Ensure JAX and dependencies are installed

✨ Features:
   ✓ Real-time game visualization
   ✓ Cooperative gameplay with AI
   ✓ Automatic data collection
   ✓ Multiple simultaneous players
   ✓ Responsive web design
   ✓ Easy to extend and customize

═══════════════════════════════════════════════════════════════
""")
